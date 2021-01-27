from collections import OrderedDict, namedtuple
from typing import Tuple

import gtimer as gt
import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossStatistics
from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel
from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
from rlkit.torch.model_based.dreamer.utils import FreezeParameters, lambda_return

try:
    import apex
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

Plan2ExploreLosses = namedtuple(
    "Plan2ExploreLosses",
    "actor_loss vf_loss world_model_loss one_step_ensemble_loss exploration_actor_loss exploration_vf_loss",
)


class Plan2ExploreTrainer(DreamerV2Trainer):
    def __init__(
        self,
        env,
        actor,
        vf,
        world_model,
        imagination_horizon,
        image_shape,
        exploration_actor,
        exploration_vf,
        one_step_ensemble,
        target_vf,
        exploration_target_vf,
        discount=0.99,
        reward_scale=1.0,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=6e-4,
        optimizer_class="torch_adam",
        use_amp=False,
        opt_level="O1",
        lam=0.95,
        free_nats=3.0,
        kl_loss_scale=1.0,
        pred_discount_loss_scale=10.0,
        adam_eps=1e-7,
        weight_decay=0.0,
        debug=False,
        exploration_reward_scale=10000,
        image_goals=None,
        policy_gradient_loss_scale=0.0,
        actor_entropy_loss_schedule="0.0",
        use_ppo_loss=False,
        ppo_clip_param=0.2,
        num_actor_value_updates=1,
        train_exploration_actor_with_intrinsic_and_extrinsic_reward=False,
        train_actor_with_intrinsic_and_extrinsic_reward=False,
        detach_rewards=True,
        image_goals_path=None,
        state_loss_scale=0,
        train_decoder_on_second_output_only=False,
        use_next_feat_for_computing_reward=False,
        one_step_ensemble_pred_prior_from_prior=True,
    ):
        super(Plan2ExploreTrainer, self).__init__(
            env,
            actor,
            vf,
            world_model,
            imagination_horizon,
            image_shape,
            target_vf=target_vf,
            discount=discount,
            reward_scale=reward_scale,
            actor_lr=actor_lr,
            vf_lr=vf_lr,
            world_model_lr=world_model_lr,
            optimizer_class=optimizer_class,
            use_amp=use_amp,
            opt_level=opt_level,
            lam=lam,
            free_nats=free_nats,
            kl_loss_scale=kl_loss_scale,
            pred_discount_loss_scale=pred_discount_loss_scale,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            debug=debug,
            image_loss_scale=1.0,
            reward_loss_scale=1.0,
            transition_loss_scale=0.0,
            entropy_loss_scale=0.0,
            forward_kl=True,
            policy_gradient_loss_scale=policy_gradient_loss_scale,
            actor_entropy_loss_schedule=actor_entropy_loss_schedule,
            soft_target_tau=1,
            target_update_period=1,
            initialize_amp=False,
            use_ppo_loss=use_ppo_loss,
            ppo_clip_param=ppo_clip_param,
            num_actor_value_updates=num_actor_value_updates,
            detach_rewards=detach_rewards,
            state_loss_scale=state_loss_scale,
            train_decoder_on_second_output_only=train_decoder_on_second_output_only,
        )
        if image_goals_path:
            self.image_goals = np.load(image_goals_path)
        else:
            self.image_goals = None
        self.exploration_actor = exploration_actor.to(ptu.device)
        self.exploration_vf = exploration_vf.to(ptu.device)
        self.exploration_target_vf = exploration_target_vf.to(ptu.device)
        self.one_step_ensemble = one_step_ensemble.to(ptu.device)

        self.one_step_ensemble_optimizer = self.optimizer_class(
            self.one_step_ensemble.parameters(),
            lr=world_model_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.exploration_actor_optimizer = self.optimizer_class(
            self.exploration_actor.parameters(),
            lr=actor_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.exploration_vf_optimizer = self.optimizer_class(
            self.exploration_vf.parameters(),
            lr=vf_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        if self.use_amp:
            models, optimizers = amp.initialize(
                [
                    self.world_model.action_step_feature_extractor,
                    self.world_model.action_step_mlp,
                    self.world_model.obs_step_mlp,
                    self.world_model.conv_decoder,
                    self.world_model.conv_encoder,
                    self.world_model.pred_discount,
                    self.world_model.reward,
                    self.world_model.rnn,
                    self.actor,
                    self.vf,
                    self.one_step_ensemble,
                    self.exploration_actor,
                    self.exploration_vf,
                ],
                [
                    self.world_model_optimizer,
                    self.actor_optimizer,
                    self.vf_optimizer,
                    self.one_step_ensemble_optimizer,
                    self.exploration_actor_optimizer,
                    self.exploration_vf_optimizer,
                ],
                opt_level=opt_level,
                num_losses=6,
            )
            (
                self.world_model.action_step_feature_extractor,
                self.world_model.action_step_mlp,
                self.world_model.obs_step_mlp,
                self.world_model.conv_decoder,
                self.world_model.conv_encoder,
                self.world_model.pred_discount,
                self.world_model.reward,
                self.world_model.rnn,
                self.actor,
                self.vf,
                self.one_step_ensemble,
                self.exploration_actor,
                self.exploration_vf,
            ) = models
            (
                self.world_model_optimizer,
                self.actor_optimizer,
                self.vf_optimizer,
                self.one_step_ensemble_optimizer,
                self.exploration_actor_optimizer,
                self.exploration_vf_optimizer,
            ) = optimizers

        self.exploration_reward_scale = exploration_reward_scale
        self.train_exploration_actor_with_intrinsic_and_extrinsic_reward = (
            train_exploration_actor_with_intrinsic_and_extrinsic_reward
        )
        self.train_actor_with_intrinsic_and_extrinsic_reward = (
            train_actor_with_intrinsic_and_extrinsic_reward
        )
        self.use_next_feat_for_computing_reward = use_next_feat_for_computing_reward
        self.one_step_ensemble_pred_prior_from_prior = (
            one_step_ensemble_pred_prior_from_prior
        )

    def try_update_target_networks(self):
        if (
            self.target_vf
            and self.exploration_target_vf
            and self._n_train_steps_total % self.target_update_period == 0
        ):
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
        ptu.soft_update_from_to(
            self.exploration_vf, self.exploration_target_vf, self.soft_target_tau
        )

    def compute_exploration_reward(
        self, exploration_imag_deter_states, exploration_imag_actions
    ):
        pred_embeddings = []
        input_state = exploration_imag_deter_states
        exploration_imag_actions = torch.cat(
            [
                exploration_imag_actions[i, :, :]
                for i in range(exploration_imag_actions.shape[0])
            ]
        )
        for mdl in range(self.one_step_ensemble.num_models):
            inputs = torch.cat((input_state, exploration_imag_actions), 1)
            pred_embeddings.append(
                self.one_step_ensemble.forward_ith_model(inputs, mdl).mean.unsqueeze(0)
            )
        pred_embeddings = torch.cat(pred_embeddings)

        assert pred_embeddings.shape[0] == self.one_step_ensemble.num_models
        assert pred_embeddings.shape[1] == input_state.shape[0]
        assert len(pred_embeddings.shape) == 3
        # computes std across ensembles, squares it to compute variance and then computes the mean across the vector dim
        reward = (pred_embeddings.std(dim=0) * pred_embeddings.std(dim=0)).mean(dim=1)
        return reward

    def imagine_ahead(self, state, actor):
        new_state = {}
        for k, v in state.items():
            with torch.no_grad():
                v = v[:, :-1]  # avoid imagining forward from terminal states
                new_state[k] = torch.cat([v[:, i, :] for i in range(v.shape[1])])
        if self.image_goals is not None:
            image_goals = ptu.from_numpy(
                self.image_goals[
                    np.random.choice(
                        range(len(self.image_goals)), size=(new_state["stoch"].shape[0])
                    )
                ]
            )
            init_state = self.world_model.initial(new_state["stoch"].shape[0])
            feat = self.world_model.get_feat(init_state).detach()
            action = actor(feat).rsample().detach()
            action = ptu.zeros_like(action)  # ensures it is a dummy action
            encoded_image_goals = self.world_model.encode(
                image_goals.flatten(start_dim=1, end_dim=3)
            )
            post_params, _, _, _, _ = self.world_model.forward_batch(
                encoded_image_goals, action, init_state
            )
            featurized_image_goals = self.world_model.get_feat(post_params).detach()
            rewards = []

        feats = []
        actions = []
        log_probs = []
        states = []
        next_states = []
        next_feats = []
        for _ in range(self.imagination_horizon):
            feat = self.world_model.get_feat(new_state)
            states.append(new_state["deter"])
            action_dist = actor(feat.detach())
            if type(actor) == ConditionalActorModel:
                action, log_prob = action_dist.rsample_and_log_prob()
            else:
                action = action_dist.rsample()
                log_prob = action_dist.log_prob(action)
            new_state = self.world_model.action_step(new_state, action)
            new_feat = self.world_model.get_feat(new_state)
            next_states.append(new_state["deter"])

            feats.append(feat.unsqueeze(0))
            next_feats.append(new_feat.unsqueeze(0))
            actions.append(action.unsqueeze(0))
            log_probs.append(log_prob.unsqueeze(0))
            if self.image_goals is not None:
                reward = -1 * torch.linalg.norm(
                    featurized_image_goals - self.world_model.get_feat(new_state), dim=1
                ).unsqueeze(0)
                rewards.append(reward)

        feats = torch.cat(feats)
        next_feats = torch.cat(next_feats)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        if self.image_goals is not None:
            rewards = torch.cat(rewards).unsqueeze(-1)
            return feats, next_feats, actions, log_probs, states, next_states, rewards
        return feats, next_feats, actions, log_probs, states, next_states

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
        **kwargs,
    ) -> Tuple[Plan2ExploreLosses, LossStatistics]:
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        """
        World Model Loss
        """
        (
            post,
            prior,
            post_dist,
            prior_dist,
            image_dist,
            reward_dist,
            pred_discount_dist,
            embed,
        ) = self.world_model(obs, actions)
        # stack obs, rewards and terminals along path dimension
        obs = torch.cat([obs[:, i, :] for i in range(obs.shape[1])])
        rewards = torch.cat([rewards[:, i, :] for i in range(rewards.shape[1])])
        terminals = torch.cat([terminals[:, i, :] for i in range(terminals.shape[1])])
        actions = torch.cat([actions[:, i, :] for i in range(actions.shape[1])])
        embed = torch.cat([embed[:, i, :] for i in range(embed.shape[1])])
        if self.one_step_ensemble_pred_prior_from_prior:
            deter = torch.cat(
                [prior["deter"][:, i, :] for i in range(prior["deter"].shape[1])]
            )
            prev_deter = torch.cat(
                (ptu.zeros_like(prior["deter"][:, 0:1, :]), prior["deter"][:, :-1, :]),
                dim=1,
            )
            prev_deter = torch.cat(
                [prev_deter[:, i, :] for i in range(prev_deter.shape[1])]
            )
        else:
            prev_deter = post["deter"][:, :-1, :]
            prev_deter = torch.cat(
                [prev_deter[:, i, :] for i in range(prev_deter.shape[1])]
            )
            deter = prior["deter"][:, 1:, :]
            deter = torch.cat([deter[:, i, :] for i in range(deter.shape[1])])

        (
            world_model_loss,
            div,
            image_pred_loss,
            reward_pred_loss,
            transition_loss,
            entropy_loss,
            pred_discount_loss,
        ) = self.world_model_loss(
            image_dist,
            reward_dist,
            prior,
            post,
            prior_dist,
            post_dist,
            pred_discount_dist,
            obs,
            rewards,
            terminals,
        )

        self.update_network(
            self.world_model,
            self.world_model_optimizer,
            world_model_loss,
            0,
            self.world_model_gradient_clip,
        )

        """
        Actor Loss
        """
        world_model_params = list(self.world_model.parameters())
        vf_params = list(self.vf.parameters())
        target_vf_params = list(self.target_vf.parameters())
        one_step_ensemble_params = list(self.one_step_ensemble.parameters())
        exploration_vf_params = list(self.exploration_vf.parameters())
        exploration_target_vf_params = list(self.exploration_target_vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())

        with FreezeParameters(world_model_params):
            if self.image_goals is not None:
                (
                    imag_feat,
                    imag_next_feat,
                    imag_actions,
                    imag_log_probs,
                    imag_deter_states,
                    imag_next_deter_states,
                    extrinsic_reward,
                ) = self.imagine_ahead(post, actor=self.actor)
            else:
                (
                    imag_feat,
                    imag_next_feat,
                    imag_actions,
                    imag_log_probs,
                    imag_deter_states,
                    imag_next_deter_states,
                ) = self.imagine_ahead(post, actor=self.actor)
        with FreezeParameters(world_model_params + vf_params + target_vf_params):
            intrinsic_reward = self.compute_exploration_reward(
                imag_deter_states, imag_actions
            )
            if self.image_goals is None:
                if self.use_next_feat_for_computing_reward:
                    extrinsic_reward = self.world_model.reward(imag_next_feat)
                else:
                    extrinsic_reward = self.world_model.reward(imag_feat)

            intrinsic_reward = torch.cat(
                [
                    intrinsic_reward[i : i + imag_feat.shape[1]]
                    .unsqueeze(0)
                    .unsqueeze(2)
                    for i in range(
                        0,
                        intrinsic_reward.shape[0],
                        imag_feat.shape[1],
                    )
                ],
                0,
            )
            if self.train_actor_with_intrinsic_and_extrinsic_reward:
                imag_reward = (
                    intrinsic_reward * self.exploration_reward_scale + extrinsic_reward
                )
            else:
                imag_reward = extrinsic_reward
            with FreezeParameters(pred_discount_params):
                if self.use_next_feat_for_computing_reward:
                    discount = self.world_model.get_dist(
                        self.world_model.pred_discount(imag_next_feat),
                        std=None,
                        normal=False,
                    ).mean
                else:
                    discount = self.world_model.get_dist(
                        self.world_model.pred_discount(imag_feat),
                        std=None,
                        normal=False,
                    ).mean
            if self.use_next_feat_for_computing_reward:
                imag_target_value = self.target_vf(imag_next_feat)
                imag_value = self.vf(imag_next_feat)
            else:
                imag_target_value = self.target_vf(imag_feat)
                imag_value = self.vf(imag_feat)
        imag_returns = lambda_return(
            imag_reward[:-1],
            imag_target_value[:-1],
            discount[:-1],
            bootstrap=imag_target_value[-1],
            lambda_=self.lam,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()[:-1]
        (
            actor_loss,
            dynamics_backprop_loss,
            policy_gradient_loss,
            actor_entropy_loss,
            actor_entropy_loss_scale,
            log_probs,
        ) = (0, 0, 0, 0, 0, 0)
        for _ in range(self.num_actor_value_updates):
            (
                actor_loss_,
                dynamics_backprop_loss_,
                policy_gradient_loss_,
                actor_entropy_loss_,
                actor_entropy_loss_scale_,
                log_probs_,
            ) = self.actor_loss(
                imag_returns,
                imag_value,
                imag_feat,
                imag_actions,
                weights,
                imag_log_probs,
                self.actor,
            )
            self.update_network(
                self.actor,
                self.actor_optimizer,
                actor_loss_,
                1,
                self.actor_gradient_clip,
            )
            actor_loss += actor_loss_.item()
            dynamics_backprop_loss += dynamics_backprop_loss_.item()
            policy_gradient_loss += policy_gradient_loss_.item()
            actor_entropy_loss += actor_entropy_loss_.item()
            actor_entropy_loss_scale += actor_entropy_loss_scale_
            log_probs += log_probs_.item()

        actor_loss /= self.num_actor_value_updates
        dynamics_backprop_loss /= self.num_actor_value_updates
        policy_gradient_loss /= self.num_actor_value_updates
        actor_entropy_loss /= self.num_actor_value_updates
        actor_entropy_loss_scale /= self.num_actor_value_updates
        log_probs /= self.num_actor_value_updates
        """
        Value Loss
        """
        with torch.no_grad():
            imag_feat_v = imag_feat.detach()
            imag_next_feat_v = imag_next_feat.detach()
            target = imag_returns.detach()
            weights = weights.detach()

        if self.use_next_feat_for_computing_reward:
            vf_loss, imag_value_mean = self.value_loss(
                imag_next_feat_v, weights, target, vf=self.vf
            )
        else:
            vf_loss, imag_value_mean = self.value_loss(
                imag_feat_v, weights, target, vf=self.vf
            )

        self.update_network(
            self.vf, self.vf_optimizer, vf_loss, 2, self.value_gradient_clip
        )

        """
        One Step Ensemble Loss
        """
        batch_size = rewards.shape[0]
        bagging_size = int(1 * batch_size)
        indices = np.random.randint(
            low=0,
            high=batch_size,
            size=[self.one_step_ensemble.num_models, bagging_size],
        )
        ensemble_loss = 0
        for mdl in range(self.one_step_ensemble.num_models):
            mdl_actions = actions[indices[mdl, :]].detach()
            if self.one_step_ensemble.output_embeddings:
                input_state = deter[indices[mdl, :]].detach()
                target_prediction = embed[indices[mdl, :]].detach()
            else:
                input_state = prev_deter[indices[mdl, :]].detach()
                target_prediction = deter[indices[mdl, :]].detach()
            inputs = torch.cat((input_state, mdl_actions), 1)
            member_pred = self.one_step_ensemble.forward_ith_model(
                inputs, mdl
            )  # predict embedding of next state
            member_loss = -1 * member_pred.log_prob(target_prediction).mean()
            ensemble_loss += member_loss

        self.update_network(
            self.one_step_ensemble,
            self.one_step_ensemble_optimizer,
            ensemble_loss,
            3,
            self.world_model_gradient_clip,
        )

        """
        Exploration Actor Loss
        """
        with FreezeParameters(world_model_params):
            if self.image_goals is not None:
                (
                    exploration_imag_feat,
                    exploration_imag_next_feat,
                    exploration_imag_actions,
                    exploration_imag_log_probs,
                    exploration_imag_deter_states,
                    exploration_imag_next_deter_states,
                    exploration_extrinsic_reward,
                ) = self.imagine_ahead(
                    post,
                    actor=self.exploration_actor,
                )
            else:
                (
                    exploration_imag_feat,
                    exploration_imag_next_feat,
                    exploration_imag_actions,
                    exploration_imag_log_probs,
                    exploration_imag_deter_states,
                    exploration_imag_next_deter_states,
                ) = self.imagine_ahead(
                    post,
                    actor=self.exploration_actor,
                )
        with FreezeParameters(
            world_model_params
            + exploration_vf_params
            + one_step_ensemble_params
            + exploration_target_vf_params
        ):
            exploration_intrinsic_reward = self.compute_exploration_reward(
                exploration_imag_deter_states, exploration_imag_actions
            )
            if self.image_goals is None:
                if self.use_next_feat_for_computing_reward:
                    exploration_extrinsic_reward = self.world_model.reward(
                        exploration_imag_next_feat
                    )
                else:
                    exploration_extrinsic_reward = self.world_model.reward(
                        exploration_imag_feat
                    )

            exploration_reward = torch.cat(
                [
                    exploration_intrinsic_reward[i : i + exploration_imag_feat.shape[1]]
                    .unsqueeze(0)
                    .unsqueeze(2)
                    for i in range(
                        0,
                        exploration_intrinsic_reward.shape[0],
                        exploration_imag_feat.shape[1],
                    )
                ],
                0,
            )
            if self.train_exploration_actor_with_intrinsic_and_extrinsic_reward:
                exploration_reward = (
                    exploration_reward * self.exploration_reward_scale
                    + exploration_extrinsic_reward
                )

            with FreezeParameters(pred_discount_params):
                if self.use_next_feat_for_computing_reward:
                    exploration_discount = self.world_model.get_dist(
                        self.world_model.pred_discount(exploration_imag_next_feat),
                        std=None,
                        normal=False,
                    ).mean
                else:
                    exploration_discount = self.world_model.get_dist(
                        self.world_model.pred_discount(exploration_imag_feat),
                        std=None,
                        normal=False,
                    ).mean
            if self.use_next_feat_for_computing_reward:
                exploration_imag_target_value = self.exploration_target_vf(
                    exploration_imag_next_feat
                )
                exploration_imag_value = self.exploration_vf(exploration_imag_next_feat)
            else:
                exploration_imag_target_value = self.exploration_target_vf(
                    exploration_imag_feat
                )
                exploration_imag_value = self.exploration_vf(exploration_imag_feat)
        exploration_imag_returns = lambda_return(
            exploration_reward[:-1],
            exploration_imag_target_value[:-1],
            exploration_discount[:-1],
            bootstrap=exploration_imag_target_value[-1],
            lambda_=self.lam,
        )
        exploration_weights = torch.cumprod(
            torch.cat(
                [torch.ones_like(exploration_discount[:1]), exploration_discount[:-2]],
                0,
            ),
            0,
        ).detach()

        (
            exploration_actor_loss,
            exploration_dynamics_backprop_loss,
            exploration_policy_gradient_loss,
            exploration_actor_entropy_loss,
            exploration_actor_entropy_loss_scale,
            exploration_log_probs,
        ) = (0, 0, 0, 0, 0, 0)
        for _ in range(self.num_actor_value_updates):
            (
                exploration_actor_loss_,
                exploration_dynamics_backprop_loss_,
                exploration_policy_gradient_loss_,
                exploration_actor_entropy_loss_,
                exploration_actor_entropy_loss_scale_,
                exploration_log_probs_,
            ) = self.actor_loss(
                exploration_imag_returns,
                exploration_imag_value,
                exploration_imag_feat,
                exploration_imag_actions,
                exploration_weights,
                exploration_imag_log_probs,
                self.exploration_actor,
            )
            self.update_network(
                self.exploration_actor,
                self.exploration_actor_optimizer,
                exploration_actor_loss_,
                4,
                self.actor_gradient_clip,
            )
            exploration_actor_loss += exploration_actor_loss_.item()
            exploration_dynamics_backprop_loss += (
                exploration_dynamics_backprop_loss_.item()
            )
            exploration_policy_gradient_loss += exploration_policy_gradient_loss_.item()
            exploration_actor_entropy_loss += exploration_actor_entropy_loss_.item()
            exploration_actor_entropy_loss_scale += (
                exploration_actor_entropy_loss_scale_
            )
            exploration_log_probs += exploration_log_probs_.item()

        exploration_actor_loss /= self.num_actor_value_updates
        exploration_dynamics_backprop_loss /= self.num_actor_value_updates
        exploration_policy_gradient_loss /= self.num_actor_value_updates
        exploration_actor_entropy_loss /= self.num_actor_value_updates
        exploration_actor_entropy_loss_scale /= self.num_actor_value_updates
        exploration_log_probs /= self.num_actor_value_updates
        """
        Exploration Value Loss
        """
        with torch.no_grad():
            exploration_imag_feat_v = exploration_imag_feat.detach()
            exploration_imag_next_feat_v = exploration_imag_next_feat.detach()
            exploration_value_target = exploration_imag_returns.detach()
            exploration_weights = exploration_weights.detach()

        if self.use_next_feat_for_computing_reward:
            exploration_vf_loss, exploration_imag_value_mean = self.value_loss(
                exploration_imag_next_feat_v,
                exploration_weights,
                exploration_value_target,
                vf=self.exploration_vf,
            )
        else:
            exploration_vf_loss, exploration_imag_value_mean = self.value_loss(
                exploration_imag_feat_v,
                exploration_weights,
                exploration_value_target,
                vf=self.exploration_vf,
            )

        self.update_network(
            self.exploration_vf,
            self.exploration_vf_optimizer,
            exploration_vf_loss,
            5,
            self.value_gradient_clip,
        )
        if type(image_pred_loss) == tuple:
            image_pred_loss, state_pred_loss = image_pred_loss
            log_state_pred_loss = True
        else:
            log_state_pred_loss = False

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["Value Loss"] = vf_loss.item()
            eval_statistics["World Model Loss"] = world_model_loss.item()
            eval_statistics["Image Loss"] = image_pred_loss.item()
            if log_state_pred_loss:
                eval_statistics["State Prediction Loss"] = state_pred_loss.item()
            eval_statistics["Reward Loss"] = reward_pred_loss.item()
            eval_statistics["Divergence Loss"] = div.item()
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()

            eval_statistics["Actor Loss"] = actor_loss
            eval_statistics["Dynamics Backprop Loss"] = dynamics_backprop_loss
            eval_statistics["Reinforce Loss"] = policy_gradient_loss
            eval_statistics["Actor Entropy Loss"] = actor_entropy_loss
            eval_statistics["Actor Entropy Loss Scale"] = actor_entropy_loss_scale

            eval_statistics["Exploration Actor Loss"] = exploration_actor_loss
            eval_statistics[
                "Exploration Dynamics Backprop Loss"
            ] = exploration_dynamics_backprop_loss
            eval_statistics[
                "Exploration Reinforce Loss"
            ] = exploration_policy_gradient_loss
            eval_statistics[
                "Exploration Actor Entropy Loss"
            ] = exploration_actor_entropy_loss
            eval_statistics[
                "Exploration Actor Entropy Loss Scale"
            ] = exploration_actor_entropy_loss_scale

            eval_statistics["Imagined Returns"] = imag_returns.mean().item()
            eval_statistics["Imagined Rewards"] = imag_reward.mean().item()
            eval_statistics["Imagined Values"] = imag_value_mean.item()
            eval_statistics["Predicted Rewards"] = reward_dist.mean.mean().item()
            eval_statistics[
                "Imagined Intrinsic Rewards"
            ] = intrinsic_reward.mean().item()
            eval_statistics[
                "Imagined Extrinsic Rewards"
            ] = extrinsic_reward.mean().item()

            eval_statistics["One Step Ensemble Loss"] = ensemble_loss.item()
            eval_statistics["Exploration Value Loss"] = exploration_vf_loss.item()
            eval_statistics[
                "Exploration Imagined Values"
            ] = exploration_imag_value_mean.item()

            eval_statistics[
                "Exploration Imagined Returns"
            ] = exploration_imag_returns.mean().item()
            eval_statistics[
                "Exploration Imagined Rewards"
            ] = exploration_reward.mean().item()
            eval_statistics[
                "Exploration Imagined Intrinsic Rewards"
            ] = exploration_intrinsic_reward.mean().item()
            eval_statistics[
                "Exploration Imagined Extrinsic Rewards"
            ] = exploration_extrinsic_reward.mean().item()

        loss = Plan2ExploreLosses(
            actor_loss=actor_loss,
            world_model_loss=world_model_loss,
            vf_loss=vf_loss,
            one_step_ensemble_loss=ensemble_loss,
            exploration_actor_loss=exploration_actor_loss,
            exploration_vf_loss=exploration_vf_loss,
        )

        return loss, eval_statistics

    @property
    def networks(self):
        return [
            self.actor,
            self.vf,
            self.target_vf,
            self.world_model,
            self.one_step_ensemble,
            self.exploration_actor,
            self.exploration_vf,
            self.exploration_target_vf,
        ]

    @property
    def optimizers(self):
        return [
            self.actor_optimizer,
            self.vf_optimizer,
            self.world_model_optimizer,
            self.one_step_ensemble_optimizer,
            self.exploration_actor_optimizer,
            self.exploration_vf_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            actor=self.actor,
            world_model=self.world_model,
            vf=self.vf,
            one_step_ensemble=self.one_step_ensemble,
            exploration_actor=self.exploration_actor,
            exploration_vf=self.exploration_vf,
            exploration_target_vf=self.exploration_target_vf,
        )
