import time
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy
from rlkit.torch.model_based.plan2explore.basic_mcts_wm_expl import UCT_search


class HybridMCTSPolicy(Policy):
    """"""

    def __init__(
        self,
        world_model,
        max_steps,
        num_primitives,
        action_dim,
        action_space,
        actor,
        one_step_ensemble,
        mcts_iterations,
        exploration_weight,
        open_loop_plan=False,
        parallelize=False,
        exploration=False,
        randomly_sample_discrete_actions=False,
    ):
        self.world_model = world_model
        cont_action = action_space.low[num_primitives:]
        actions = np.array(
            [
                np.concatenate((np.squeeze(np.eye(num_primitives)[i]), cont_action))
                for i in range(num_primitives)
            ]
        )
        actions = ptu.from_numpy(actions)
        self.world_model.actions = actions
        self.max_steps = max_steps
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.mcts_iterations = mcts_iterations
        self.exploration_weight = exploration_weight
        self.open_loop_plan = open_loop_plan
        self.parallelize = parallelize
        self.actor = actor
        self.exploration = exploration
        self.one_step_ensemble = one_step_ensemble
        self.randomly_sample_discrete_actions = randomly_sample_discrete_actions

    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        observation = ptu.from_numpy(np.array(observation))
        if self.state:
            latent, action = self.state
        else:
            latent = self.world_model.initial(observation.shape[0])
            action = ptu.zeros((observation.shape[0], self.action_dim))
        if self.open_loop_plan:
            actions = [
                self.actions[i][self.ctr].reshape(1, -1)
                for i in range(observation.shape[0])
            ]
            action = torch.cat(actions)
            assert action.shape == (observation.shape[0], self.action_dim)
            if self.randomly_sample_discrete_actions:
                discrete_action = ptu.from_numpy(
                    np.eye(self.world_model.env.num_primitives)[
                        np.random.choice(
                            self.world_model.env.num_primitives,
                            observation.shape[0],
                        )
                    ]
                )
                embed = self.world_model.encode(observation)
                start_state, _ = self.world_model.obs_step(latent, action, embed)
                action_input = (discrete_action, self.world_model.get_feat(start_state))
                action_dist = self.actor(action_input)
                continuous_action = self.actor.compute_exploration_action(
                    action_dist.sample(), 0.3
                )
                action = torch.cat((discrete_action, continuous_action), 1)
        else:
            embed = self.world_model.encode(observation)
            start_state, _ = self.world_model.obs_step(latent, action, embed)
            actions = []
            start_state = (start_state, 0)
            action = UCT_search(
                self.world_model,
                self.one_step_ensemble,
                self.actor,
                start_state,
                self.mcts_iterations,
                self.world_model.env.max_steps,
                self.world_model.env.num_primitives,
                return_open_loop_plan=False,
                exploration_reward=self.exploration,
                evaluation=not self.exploration,
                exploration_weight=self.exploration_weight,
            )
        self.ctr += 1
        self.state = (latent, action)
        return ptu.get_numpy(action), {}

    def reset(self, o):
        self.state = None
        self.ctr = 0
        o = np.concatenate([o_.reshape(1, -1) for o_ in o])
        latent = self.world_model.initial(o.shape[0])
        action = ptu.zeros((o.shape[0], self.action_dim))
        o = ptu.from_numpy(np.array(o))
        embed = self.world_model.encode(o)
        start_state, _ = self.world_model.obs_step(latent, action, embed)
        if self.open_loop_plan:
            state_n = {}
            for k, v in start_state.items():
                state_n[k] = v[0:1]
            state_n = (state_n, 0)
            self.actions = UCT_search(
                self.world_model,
                self.one_step_ensemble,
                self.actor,
                state_n,
                self.mcts_iterations,
                self.world_model.env.max_steps,
                self.world_model.env.num_primitives,
                return_open_loop_plan=True,
                exploration_reward=self.exploration,
                return_top_k_paths=True,
                k=o.shape[0],
                evaluation=not self.exploration,
                exploration_weight=self.exploration_weight,
            )


class ActionSpaceSamplePolicy(Policy):
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return (
            np.array([self.env.action_space.sample() for _ in range(self.env.n_envs)]),
            {},
        )

    def reset(self, o):
        return super().reset()
