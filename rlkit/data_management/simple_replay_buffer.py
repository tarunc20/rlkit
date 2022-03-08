import warnings
from collections import OrderedDict

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace=True,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        env_info,
        **kwargs
    ):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(
            self._size,
            size=batch_size,
            replace=self._replace or self._size < batch_size,
        )
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
            )
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {key: self._env_infos[key][idx] for key in self._env_info_keys}

    def batch_env_info_dict(self, indices):
        return {key: self._env_infos[key][indices] for key in self._env_info_keys}

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([("size", self._size)])


class ImageReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        image_flattened_shape,
        env_info_sizes,
        replace=True,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self.image_flattened_shape = image_flattened_shape
        self._image_observations = np.zeros(
            (max_replay_buffer_size, image_flattened_shape), dtype=np.uint8
        )
        self._state_observations = np.zeros(
            (max_replay_buffer_size, observation_dim - image_flattened_shape),
        )
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_image_observations = np.zeros(
            (max_replay_buffer_size, image_flattened_shape), dtype=np.uint8
        )
        self._next_state_observations = np.zeros(
            (max_replay_buffer_size, observation_dim - image_flattened_shape)
        )
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        env_info,
        **kwargs
    ):
        self._image_observations[self._top] = observation[: self.image_flattened_shape]
        self._state_observations[self._top] = observation[self.image_flattened_shape :]
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_image_observations[self._top] = next_observation[
            : self.image_flattened_shape
        ]
        self._next_state_observations[self._top] = next_observation[
            self.image_flattened_shape :
        ]

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def random_batch(self, batch_size):
        indices = np.random.choice(
            self._size,
            size=batch_size,
            replace=self._replace or self._size < batch_size,
        )
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
            )
        batch = dict(
            observations=np.concatenate(
                (self._image_observations[indices], self._state_observations[indices]),
                axis=1,
            ),
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=np.concatenate(
                (
                    self._next_image_observations[indices],
                    self._next_state_observations[indices],
                ),
                axis=1,
            ),
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch
