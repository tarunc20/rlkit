from collections import OrderedDict, namedtuple
from typing import Tuple

import gtimer as gt
import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossFunction, LossStatistics
from rlkit.torch.model_based.dreamer.utils import (
    FreezeParameters,
    zero_grad,
    lambda_return,
)
from rlkit.torch.torch_rl_algorithm import TorchTrainer

try:
    import apex
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

Plan2ExploreLosses = namedtuple(
    "Plan2ExploreLosses",
    "actor_loss vf_loss world_model_loss",
)


class Plan2ExploreTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        env,
        actor,
        vf,
        world_model,
        exploration_actor,
        exploration_vf,
        one_step_ensemble,
        discount=0.99,
        reward_scale=1.0,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=6e-4,
        optimizer_class="torch_adam",
        use_amp=False,
        opt_level="O1",
        gradient_clip=100.0,
        lam=0.95,
        imagination_horizon=4,
        free_nats=3.0,
        kl_loss_scale=1.0,
        pcont_loss_scale=10.0,
        adam_eps=1e-7,
        weight_decay=0.0,
        use_pcont=True,
        plotter=None,
        render_eval_paths=False,
        debug=False,
        intrinsic_reward_scale=10000,
    ):
        super().__init__()

        torch.autograd.set_detect_anomaly(debug)

        torch.backends.cudnn.benchmark = True

        self.env = env
        self.actor = actor.to(ptu.device)
        self.world_model = world_model.to(ptu.device)
        self.vf = vf.to(ptu.device)

        self.exploration_actor = exploration_actor.to(ptu.device)
        self.exploration_vf = exploration_vf.to(ptu.device)
        self.one_step_ensemble = one_step_ensemble.to(ptu.device)

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        if optimizer_class == "torch_adam":
            optimizer_class = optim.Adam
        elif optimizer_class == "apex_adam" and APEX_AVAILABLE:
            optimizer_class = apex.optimizers.FusedAdam

        self.actor_optimizer = optimizer_class(
            self.actor.parameters(),
            lr=actor_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.world_model_optimizer = optimizer_class(
            self.world_model.parameters(),
            lr=world_model_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.one_step_ensemble_optimizer = optimizer_class(
            self.one_step_ensemble.parameters(),
            lr=world_model_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.exploration_actor_optimizer = optimizer_class(
            self.exploration_actor.parameters(),
            lr=actor_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.exploration_vf_optimizer = optimizer_class(
            self.exploration_vf.parameters(),
            lr=vf_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp:
            models, optimizers = amp.initialize(
                [self.world_model, self.actor, self.vf],
                [self.world_model_optimizer, self.actor_optimizer, self.vf_optimizer],
                opt_level=opt_level,
            )
            self.world_model, self.actor, self.vf = models
            (
                self.world_model_optimizer,
                self.actor_optimizer,
                self.vf_optimizer,
            ) = optimizers

        self.opt_level = opt_level
        self.discount = discount
        self.reward_scale = reward_scale
        self.gradient_clip = gradient_clip
        self.lam = lam
        self.imagination_horizon = imagination_horizon
        self.free_nats = free_nats
        self.kl_loss_scale = kl_loss_scale
        self.pcont_loss_scale = pcont_loss_scale
        self.use_pcont = use_pcont
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """

        self._n_train_steps_total += 1
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp("Plan2Explore training", unique=False)

    def compute_intrinsic_reward(
        self, exploration_imag_states, exploration_imag_actions
    ):
        pred_embeddings = []
        input_state = exploration_imag_states["deter"]
        actions = exploration_imag_actions
        for mdl in range(self.one_step_ensemble.num_models):
            pred_embeddings.append(
                self.one_step_ensemble[mdl](input_state, actions).mean
            )
        pred_embeddings = torch.cat(pred_embeddings)
        assert pred_embeddings.shape == (
            self.one_step_ensemble.num_models,
            input_state.shape[0],
            input_state.shape[1],
        ), pred_embeddings.shape
        reward = (pred_embeddings.std(dim=0) * pred_embeddings.std(dim=0)).mean(
            dim=1
        ) * self.intrinsic_reward_scale
        return reward

    def imagine_ahead(self, state, actor):
        new_state = {}
        for k, v in state.items():
            with torch.no_grad():
                if self.use_pcont:  # Last step could be terminal.
                    v = v[:, :-1]
                new_state[k] = torch.cat([v[:, i, :] for i in range(v.shape[1])])
        feats = []
        actions = []
        for i in range(self.imagination_horizon):
            feat = self.world_model.get_feat(new_state).detach()
            action = actor(feat).rsample()
            new_state = self.world_model.img_step(new_state, action)
            feats.append(self.world_model.get_feat(new_state).unsqueeze(0))
            if i < self.imagination_horizon - 1:
                actions.append(action)
        feats = torch.cat(feats)
        actions = torch.cat(actions)
        return feats, actions

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
            pcont_dist,
        ) = self.world_model(obs, actions)

        # stack obs, rewards and terminals along path dimension
        obs = torch.cat([obs[:, i, :] for i in range(obs.shape[1])])
        rewards = torch.cat([rewards[:, i, :] for i in range(rewards.shape[1])])
        terminals = torch.cat([terminals[:, i, :] for i in range(terminals.shape[1])])

        image_pred_loss = (
            -1
            * image_dist.log_prob(
                self.world_model.preprocess(obs).reshape(-1, 3, 64, 64)
            ).mean()
        )
        reward_pred_loss = (
            -1 * reward_dist.log_prob(rewards.detach()).mean()
        )  # in plan2explore we only update the reward head do NOT propagate the reward back into the world model
        pcont_target = self.discount * (1 - terminals.float())
        pcont_loss = -1 * pcont_dist.log_prob(pcont_target).mean()
        div = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
        div = torch.max(div, ptu.from_numpy(np.array(self.free_nats)))
        world_model_loss = self.kl_loss_scale * div + image_pred_loss + reward_pred_loss

        if self.use_pcont:
            world_model_loss += self.pcont_loss_scale * pcont_loss

        zero_grad(self.world_model)
        if self.use_amp:
            with amp.scale_loss(
                world_model_loss, self.world_model_optimizer
            ) as scaled_world_model_loss:
                scaled_world_model_loss.backward()
        else:
            world_model_loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.world_model_optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        self.world_model_optimizer.step()

        """
        Actor Loss
        """
        with FreezeParameters(self.world_model.modules):
            imag_feat, imag_actions = self.imagine_ahead(post, actor=self.actor)
        with FreezeParameters(self.world_model.modules + self.vf.modules):
            imag_reward = self.world_model.reward(imag_feat)
            if self.use_pcont:
                with FreezeParameters([self.world_model.pcont]):
                    discount = self.world_model.get_dist(
                        self.world_model.pcont(imag_feat), std=None, normal=False
                    ).mean
            else:
                discount = self.discount * torch.ones_like(imag_reward)
            value = self.vf(imag_feat)
        imag_returns = lambda_return(
            imag_reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self.lam,
        )
        discount = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-2]], 0), 0
        ).detach()
        dynamics_backprop_loss = -(discount * imag_returns).mean()

        actor_loss = dynamics_backprop_loss

        zero_grad(self.actor)
        if self.use_amp:
            with amp.scale_loss(actor_loss, self.actor_optimizer) as scaled_actor_loss:
                scaled_actor_loss.backward()
        else:
            actor_loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.actor_optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        self.actor_optimizer.step()

        """
        Value Loss
        """
        with torch.no_grad():
            imag_feat_v = imag_feat.detach()
            target = imag_returns.detach()
            discount = discount.detach()

        value_dist = self.world_model.get_dist(self.vf(imag_feat_v)[:-1], 1)
        vf_loss = -(discount.squeeze(-1) * value_dist.log_prob(target)).mean()

        zero_grad(self.vf)
        if self.use_amp:
            with amp.scale_loss(vf_loss, self.vf_optimizer) as scaled_vf_loss:
                scaled_vf_loss.backward()
        else:
            vf_loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    self.vf.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.vf_optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        self.vf_optimizer.step()

        """
        One Step Ensemble Loss
        """
        batch_size = rewards.shape[0]
        bagging_size = int(1 * batch_size)
        indices = np.random.uniform(
            low=0,
            high=batch_size,
            size=[self.one_step_ensemble.num_models, bagging_size],
        )
        ensemble_member_losses = []
        embed = None  # world model needs to return this
        for mdl in range(self.one_step_ensemble.num_models):
            actions = actions[indices[mdl, :]]
            target_prediction = embed[indices[mdl, :]].detach()
            input_state = prior["deter"][indices[mdl, :]]
            member_pred = self.one_step_ensemble[mdl](input_state, actions)
            member_loss = member_pred.log_prob(target_prediction).mean()
            ensemble_member_losses.append(member_loss)

        ensemble_loss = -sum(ensemble_member_losses)
        zero_grad(self.one_step_ensemble)
        if self.use_amp:
            with amp.scale_loss(
                ensemble_loss, self.one_step_ensemble_optimizer
            ) as scaled_ensemble_loss:
                scaled_ensemble_loss.backward()
        else:
            ensemble_loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    self.one_step_ensemble.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.one_step_ensemble_optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        self.one_step_ensemble_optimizer.step()

        """
        Exploration Actor Loss
        """
        with FreezeParameters(self.world_model.modules):
            exploration_imag_states, exploration_imag_actions = self.imagine_ahead(
                post,
                actor=self.exploration_actor,
            )
        with FreezeParameters(self.world_model.modules + self.vf.modules):
            exploration_imag_feat = self.world_model.get_feat(exploration_imag_states)
            intrinsic_reward = self.compute_intrinsic_reward(
                exploration_imag_states, exploration_imag_actions, exploration_imag_feat
            )  # Compute Intrinsic Reward
            if self.use_pcont:
                with FreezeParameters([self.world_model.pcont]):
                    exploration_discount = self.world_model.get_dist(
                        self.world_model.pcont(exploration_imag_feat),
                        std=None,
                        normal=False,
                    ).mean
            else:
                exploration_discount = self.discount * torch.ones_like(intrinsic_reward)
            exploration_value = self.exploration_vf(exploration_imag_feat).mode()
        exploration_returns = lambda_return(
            intrinsic_reward[:-1],
            exploration_value[:-1],
            exploration_discount[:-1],
            bootstrap=exploration_value[-1],
            lambda_=self.lam,
        )
        exploration_discount = torch.cumprod(
            torch.cat(
                [torch.ones_like(exploration_discount[:1]), exploration_discount[:-2]],
                0,
            ),
            0,
        ).detach()
        exploration_actor_loss = -(exploration_discount * exploration_returns).mean()

        zero_grad(self.exploration_actor)
        if self.use_amp:
            with amp.scale_loss(
                exploration_actor_loss, self.exploration_actor_optimizer
            ) as scaled_exploration_actor_loss:
                scaled_exploration_actor_loss.backward()
        else:
            exploration_actor_loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    self.exploration_actor.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.exploration_actor_optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        self.exploration_actor_optimizer.step()
        """
        Exploration Value Loss
        """
        with torch.no_grad():
            exploration_imag_feat_v = exploration_imag_feat.detach()
            exploration_value_target = exploration_returns.detach()
            exploration_discount = exploration_discount.detach()

        exploration_value_dist = self.world_model.get_dist(
            self.exploration_vf(exploration_imag_feat_v)[:-1], 1
        )
        exploration_vf_loss = -(
            exploration_discount.squeeze(-1)
            * exploration_value_dist.log_prob(exploration_value_target)
        ).mean()

        zero_grad(self.exploration_vf)
        if self.use_amp:
            with amp.scale_loss(
                exploration_vf_loss, self.exploration_vf_optimizer
            ) as scaled_exploration_vf_loss:
                scaled_exploration_vf_loss.backward()
        else:
            exploration_vf_loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    self.exploration_vf.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.exploration_vf_optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        self.exploration_vf_optimizer.step()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["Value Loss"] = vf_loss.item()
            eval_statistics["Actor Loss"] = actor_loss.item()
            eval_statistics["World Model Loss"] = world_model_loss.item()
            eval_statistics["Image Loss"] = image_pred_loss.item()
            eval_statistics["Reward Loss"] = reward_pred_loss.item()
            eval_statistics["Divergence Loss"] = div.item()
            if self.use_pcont:
                eval_statistics["Pcont Loss"] = pcont_loss.item()
            eval_statistics["Imagined Returns"] = imag_returns.mean().item()
            eval_statistics["Imagined Rewards"] = imag_reward.mean().item()
            eval_statistics["Imagined Values"] = value_dist.mean.mean().item()
            eval_statistics["Predicted Rewards"] = reward_dist.mean.mean().item()

        loss = Plan2ExploreLosses(
            actor_loss=actor_loss,
            world_model_loss=world_model_loss,
            vf_loss=vf_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.actor,
            self.vf,
            self.world_model,
        ]

    @property
    def optimizers(self):
        return [
            self.actor_optimizer,
            self.vf_optimizer,
            self.world_model_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            actor=self.actor,
            world_model=self.world_model,
            vf=self.vf,
        )
