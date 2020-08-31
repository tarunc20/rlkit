import numpy as np
import warnings
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim


class EpisodeReplayBuffer(SimpleReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        max_path_length,
        observation_dim,
        action_dim,
        replace = True,
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self._observation_dim = get_dim(self._ob_space)
        self._action_dim = get_dim(self._action_space)
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, max_path_length, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, max_path_length, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, max_path_length, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, max_path_length, 1), dtype='uint8')
        self._replace = replace

        self._top = 0
        self._size = 0

    def add_path(self, path):
        self._observations[self._top] = path["observations"]
        self._actions[self._top] = path["actions"]
        self._rewards[self._top] = path["rewards"]
        self._terminals[self._top] = path["terminals"]
        self._next_obs[self._top] = path["next_observations"]

        self._advance()

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        return batch