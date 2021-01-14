# https://www.moderndescartes.com/essays/deep_dive_mcts/
# https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenKettleV0,
    KitchenSlideCabinetV0,
)

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.world_models import WorldModel
from rlkit.torch.model_based.plan2explore.actor_models import (
    ConditionalContinuousActorModel,
)
from rlkit.torch.model_based.plan2explore.latent_space_models import (
    OneStepEnsembleModel,
)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def compute_exploration_reward(
    one_step_ensemble, exploration_imag_deter_states, exploration_imag_actions
):
    pred_embeddings = []
    input_state = exploration_imag_deter_states
    for mdl in range(one_step_ensemble.num_models):
        inputs = torch.cat((input_state, exploration_imag_actions), 1)
        pred_embeddings.append(
            one_step_ensemble.forward_ith_model(inputs, mdl).mean.unsqueeze(0)
        )
    pred_embeddings = torch.cat(pred_embeddings)
    assert pred_embeddings.shape[0] == one_step_ensemble.num_models
    assert pred_embeddings.shape[1] == input_state.shape[0]
    assert len(pred_embeddings.shape) == 3

    reward = (pred_embeddings.std(dim=0) * pred_embeddings.std(dim=0)).mean(dim=1)
    assert (reward >= 0.0).all()
    return reward


def compute_all_paths(cur):
    if len(cur.children) == 0:
        return []
    else:
        new_l = []
        for a, child in cur.children.items():
            l = compute_all_paths(child)
            if len(l) == 0:
                new_l.append([[a, child.number_visits]])
            else:
                for path in l:
                    total_visits = path[0][1] + child.number_visits
                    p = [(a, total_visits)] + path
                    new_l.append(p)
        return new_l


def compute_top_k_paths(l, k):
    l.sort(key=lambda x: x[0][1])
    top_k_paths = l[-k:]
    actions = []
    for path in top_k_paths:
        path_as = []
        for val in path:
            a, v = val
            path_as.append(ptu.from_numpy(np.array(a).reshape(1, -1)))
        path_as = torch.cat(path_as)
        actions.append(path_as.unsqueeze(0))
    return torch.cat(actions)


@torch.no_grad()
def Advanced_UCT_search(
    wm,
    one_step_ensemble,
    actor,
    state,
    iterations,
    max_steps,
    num_primitives,
    intrinsic_vf,
    extrinsic_vf,
    evaluation=False,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
    batch_size=4,
    discount=1.0,
    dirichlet_alpha=0.03,  # todo: sweep .03, .15, .3 and other numbers
    progressive_widening_constant=0.1,
    return_open_loop_plan=True,
    return_top_k_paths=True,
    K=1,
):
    start_spot = int(1.5 * batch_size)
    root = UCTNode(
        wm,
        one_step_ensemble,
        actor,
        state,
        num_primitives,
        max_steps,
        reward=0,
        prior=0,
        step_count=0,
        parent=None,
    )
    root.expand(
        intrinsic_reward_scale=intrinsic_reward_scale,
        extrinsic_reward_scale=extrinsic_reward_scale,
        evaluation=evaluation,
    )
    root.add_exploration_noise(
        dirichlet_alpha=dirichlet_alpha, exploration_fraction=0.25
    )
    ctr = 0
    eval_list = []
    min_max_stats = MinMaxStats()
    while ctr < iterations:
        leaf = root.select_leaf(
            min_max_stats,
            discount,
            eval_list=eval_list,
            virtual_loss=ctr > start_spot,
            c1=1.25,
            c2=19652,
            progressive_widening_constant=progressive_widening_constant,
        )
        if ctr > start_spot:
            if leaf.is_terminal:
                leaf.backup(leaf.value, discount, min_max_stats, virtual_loss=True)
            else:
                eval_list.append(leaf)
                if len(eval_list) >= batch_size:
                    states = torch.cat(
                        [wm.get_feat(leaf.state) for leaf in eval_list], 0
                    )
                    values = ptu.zeros(states.shape[0])
                    if intrinsic_reward_scale > 0.0:
                        values += (
                            intrinsic_vf(states).flatten() * intrinsic_reward_scale
                        )
                    if extrinsic_reward_scale > 0.0:
                        values += (
                            extrinsic_vf(states).flatten() * extrinsic_reward_scale
                        )
                    full_state = {}
                    for leaf in eval_list:
                        for k, v in leaf.state.items():
                            if k in full_state:
                                full_state[k].append(v.repeat(num_primitives, 1))
                            else:
                                full_state[k] = [v.repeat(num_primitives, 1)]
                    full_state = {
                        k: torch.cat(full_state[k]) for k in full_state.keys()
                    }
                    discrete_actions = ptu.eye(num_primitives).repeat(len(eval_list), 1)
                    full_actions, priors = generate_full_actions(
                        wm, full_state, actor, discrete_actions, evaluation
                    )
                    new_state = wm.action_step(full_state, full_actions)
                    deter_state = full_state["deter"]
                    r = ptu.zeros(deter_state.shape[0])
                    if intrinsic_reward_scale > 0.0:
                        r += (
                            compute_exploration_reward(
                                one_step_ensemble, deter_state, full_actions
                            ).flatten()
                            * intrinsic_reward_scale
                        )
                    if extrinsic_reward_scale > 0.0:
                        r += (
                            wm.reward(wm.get_feat(new_state)).flatten()
                            * extrinsic_reward_scale
                        )
                    for i, leaf in enumerate(eval_list):
                        assert not leaf.is_terminal
                        st = {}
                        for k, v in new_state.items():
                            st[k] = v[i * num_primitives : (i + 1) * num_primitives]
                        action = full_actions[
                            i * num_primitives : (i + 1) * num_primitives
                        ]
                        reward = r[i * num_primitives : (i + 1) * num_primitives]
                        prior = priors[i * num_primitives : (i + 1) * num_primitives]
                        if not leaf.is_expanded:
                            # only leaves that are not from progressive widening
                            value = values[i].item()
                            leaf.value = value
                            leaf.backup(
                                leaf.value, discount, min_max_stats, virtual_loss=True
                            )
                        if leaf.step_count < max_steps:
                            leaf.expand_given_states_actions(st, action, prior, reward)
                        else:
                            leaf.is_expanded = True  # for terminal states
                            leaf.is_terminal = True  # for terminal states

                    eval_list = []
        else:
            if not leaf.value:
                states = wm.get_feat(leaf.state)
                value = ptu.zeros(states.shape[0])
                if intrinsic_reward_scale > 0.0:
                    value += intrinsic_vf(states)[0] * intrinsic_reward_scale
                if extrinsic_reward_scale > 0.0:
                    value += extrinsic_vf(states)[0] * extrinsic_reward_scale
                leaf.value = value.item()
            if leaf.step_count < max_steps:
                leaf.expand(
                    intrinsic_reward_scale=intrinsic_reward_scale,
                    extrinsic_reward_scale=extrinsic_reward_scale,
                    evaluation=evaluation,
                )
            else:
                leaf.is_expanded = True  # for terminal states
                leaf.is_terminal = True  # for terminal states
            leaf.backup(leaf.value, discount, min_max_stats, False)
        ctr += 1

    if return_open_loop_plan:
        if return_top_k_paths:
            l = compute_all_paths(root)
            actions = compute_top_k_paths(l, K)
            return actions
        output_actions = []
        cur = root
        while cur.children != {}:
            max_visits = -np.inf
            max_a = None
            max_child = None
            for a, child in cur.children.items():
                if child.number_visits >= max_visits:
                    max_visits = child.number_visits
                    max_a = a
                    max_child = child
            output_actions.append(np.array(max_a).reshape(1, -1))
            cur = max_child
        actions = np.concatenate(output_actions, 0)
        actions = ptu.from_numpy(actions)
        return actions
    else:
        return compute_root_action(root)


def compute_root_action(root):
    max_visit_count = -np.inf
    max_a = None
    for a, child in root.children.items():
        if child.number_visits >= max_visit_count:
            max_visit_count = child.number_visits
            max_a = a
    return ptu.from_numpy(np.array(max_a)).reshape(1, -1)


def generate_full_actions(
    wm,
    state,
    actor,
    discrete_actions,
    evaluation=False,
):
    feat = wm.get_feat(state)
    if type(actor) == ConditionalContinuousActorModel:
        action_input = (discrete_actions, feat)
        action_dist = actor(action_input)
        if evaluation:
            continuous_action = action_dist.mode()
        else:
            continuous_action = actor.compute_exploration_action(
                action_dist.sample(), 0.3
            )
        actions = torch.cat((discrete_actions, continuous_action), 1)
        priors = action_dist.log_prob(continuous_action)
    elif type(actor) == ConditionalActorModel:
        action_input = feat
        action_dist = actor(action_input)
        cont_dist = action_dist.compute_continuous_dist(discrete_actions)
        if evaluation:
            continuous_action = cont_dist.mode()
        else:
            continuous_action = actor.compute_continuous_exploration_action(
                cont_dist.sample(), 0.3
            )
        actions = torch.cat((discrete_actions, continuous_action), 1)
        priors = action_dist.log_prob_given_continuous_dist(actions, cont_dist)
    return actions, priors


def step_wm(
    wm,
    state,
    actor,
    num_primitives,
    one_step_ensemble,
    evaluation=False,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
):
    state_n = {}
    for k, v in state.items():
        state_n[k] = v.repeat(num_primitives, 1)
    discrete_actions = ptu.eye(num_primitives)
    action, priors = generate_full_actions(
        wm, state_n, actor, discrete_actions, evaluation
    )
    new_state = wm.action_step(state_n, action)
    deter_state = state_n["deter"]
    r = ptu.zeros(deter_state.shape[0])
    if intrinsic_reward_scale > 0.0:
        r += (
            compute_exploration_reward(one_step_ensemble, deter_state, action).flatten()
            * intrinsic_reward_scale
        )
    if extrinsic_reward_scale > 0.0:
        r += wm.reward(wm.get_feat(new_state)).flatten() * extrinsic_reward_scale
    return new_state, action, priors, r


class UCTNode:
    def __init__(
        self,
        wm,
        one_step_ensemble,
        actor,
        state,
        num_primitives,
        max_steps,
        reward,
        prior,
        step_count=0,
        parent=None,
    ):
        self.wm = wm
        self.one_step_ensemble = one_step_ensemble
        self.actor = actor
        self.state = state
        self.reward = reward
        self.parent = parent
        self.num_primitives = num_primitives
        self.step_count = step_count
        self.max_steps = max_steps
        self.is_expanded = False
        self.children = {}
        self.total_value = 0
        self.number_visits = 0
        self.is_terminal = False
        self.value = None
        self.prior = prior
        self.child_priors_sum = 0

    def average_value(self):
        if self.number_visits > 0:
            value = self.total_value / (self.number_visits)
        else:
            value = 0
        return value

    def Q(self, min_max_stats, discount):
        return self.reward + discount * self.average_value()

    def U(self, c1, c2):
        prior_score = math.log((self.parent.number_visits + c2 + 1) / c2) + c1
        prior_score *= math.sqrt(self.parent.number_visits) / (self.number_visits + 1)
        prior = self.prior / self.parent.child_priors_sum
        prior_score *= prior
        return prior_score

    def score(self, node, min_max_stats, discount, c1, c2):
        q = node.Q(min_max_stats, discount)
        u = node.U(c1, c2)
        return q + u

    def best_child(self, min_max_stats, discount, c1, c2):
        max_score = -np.inf
        best_node = None
        for node in self.children.values():
            score = self.score(node, min_max_stats, discount, c1, c2)
            if score > max_score:
                max_score = score
                best_node = node
        return best_node

    def select_leaf(
        self,
        min_max_stats,
        discount,
        c1,
        c2,
        eval_list,
        progressive_widening_constant,
        virtual_loss=False,
    ):
        current = self
        while current.is_expanded and not current.is_terminal:
            if virtual_loss:
                current.number_visits += 1
                current.total_value -= 10000
                min_max_stats.update(
                    current.reward + discount * current.average_value()
                )
                current.progressively_widen(eval_list, progressive_widening_constant)

            current = current.best_child(min_max_stats, discount, c1, c2)
        if virtual_loss:
            current.number_visits += 1
            current.total_value -= 10000
        return current

    def progressively_widen(self, eval_list, progressive_widening_constant):
        # progressive widening
        alpha = 0.5
        thresh = math.ceil(progressive_widening_constant * self.number_visits ** alpha)
        if len(self.children) < thresh:
            eval_list.append(self)

    def expand(self, intrinsic_reward_scale, extrinsic_reward_scale, evaluation):
        child_states, actions, priors, rewards = step_wm(
            self.wm,
            self.state,
            self.actor,
            self.num_primitives,
            self.one_step_ensemble,
            intrinsic_reward_scale=intrinsic_reward_scale,
            extrinsic_reward_scale=extrinsic_reward_scale,
            evaluation=evaluation,
        )
        self.expand_given_states_actions(child_states, actions, priors, rewards)

    def expand_given_states_actions(self, child_states, actions, priors, rewards):
        self.is_expanded = True
        self.child_priors_sum = self.child_priors_sum + priors.sum().item()
        for i in range(self.num_primitives):
            child_state = {}
            for k, v in child_states.items():
                child_state[k] = v[i : i + 1, :]
            self.add_child(
                child_state,
                actions[i, :],
                priors[i].item(),
                rewards[i].item(),
            )

    def add_child(self, state, action, prior, reward):
        key = tuple(ptu.get_numpy(action).tolist())
        node = UCTNode(
            self.wm,
            self.one_step_ensemble,
            self.actor,
            state,
            self.num_primitives,
            self.max_steps,
            reward,
            prior,
            step_count=self.step_count + 1,
            parent=self,
        )
        self.children[key] = node
        return node

    def backup(self, value, discount, min_max_stats, virtual_loss=False):
        current = self
        while current is not None:
            if virtual_loss:
                current.total_value += 10000 + value
            else:
                current.number_visits += 1
                current.total_value += value
            min_max_stats.update(current.reward + discount * current.average_value())
            value = current.reward + discount * value
            current = current.parent

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        From: https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


if __name__ == "__main__":
    env = KitchenHingeCabinetV0(
        fixed_schema=False, delta=0.0, dense=False, image_obs=True
    )
    ptu.set_gpu_mode(True)
    torch.backends.cudnn.benchmark = True

    wm = WorldModel(
        env.action_space.low.size,
        env.image_shape,
        env,
    ).to(ptu.device)
    one_step_ensemble = OneStepEnsembleModel(
        action_dim=env.action_space.low.size,
        deterministic_state_size=400,
        embedding_size=1024,
        num_models=5,
        hidden_size=400,
        num_layers=4,
        output_embeddings=False,
    ).to(ptu.device)
    actor = ConditionalContinuousActorModel(
        [400] * 4,
        wm.feature_size,
        env,
        discrete_action_dim=env.num_primitives,
        continuous_action_dim=env.max_arg_len,
    ).to(ptu.device)
    # actor = ConditionalActorModel(
    #     [400] * 4,
    #     wm.feature_size,
    #     env,
    #     discrete_continuous_dist=True,
    #     discrete_action_dim=env.num_primitives,
    #     continuous_action_dim=env.max_arg_len,
    # ).to(ptu.device)
    state = wm.initial(1)

    intrinsic_vf = Mlp(
        hidden_sizes=[400] * 3,
        output_size=1,
        input_size=wm.feature_size,
        hidden_activation=torch.nn.functional.elu,
    ).to(ptu.device)
    extrinsic_vf = Mlp(
        hidden_sizes=[400] * 3,
        output_size=1,
        input_size=wm.feature_size,
        hidden_activation=torch.nn.functional.elu,
    ).to(ptu.device)
    import time

    num_tries = 100
    for batch_size in [64, 128]:
        total_time = 0
        for i in range(num_tries):
            t = time.time()

            action = Advanced_UCT_search(
                wm,
                one_step_ensemble,
                actor,
                state,
                10000,
                env.max_steps,
                env.num_primitives,
                intrinsic_vf=intrinsic_vf,
                extrinsic_vf=extrinsic_vf,
                evaluation=True,
                intrinsic_reward_scale=0.0,
                extrinsic_reward_scale=1.0,
                batch_size=batch_size,
                start_spot=int(1.5 * batch_size),
                progressive_widening_constant=0.1,
            )
            total_time += time.time() - t
        print(batch_size, total_time / num_tries)
    # state, r, _ = step_wm(wm, one_step_ensemble, (state, 0), actor, env.num_primitives)
    # action = UCT_search(
    #     wm,
    #     one_step_ensemble,
    #     actor,
    #     (state, 0),
    #     10000,
    #     env.max_steps,
    #     env.num_primitives,
    # )[0]
    # print(action)
