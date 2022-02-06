from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim

import pdb
import warnings
import numpy as np

class EpisodeReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        env, 
        max_path_length,
        replace=True,
        batch_length=50,
        use_batch_length=False
    ):
        # Initialize the environments
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self._observation_dim = get_dim(self._ob_space)
        self._action_dim = get_dim(self._action_space)

        self.max_path_length = max_path_length
        self._max_replay_buffer_size = max_replay_buffer_size
        
        # Initialize Buffers to Store Trajectories
        self._observations = np.zeros(
            (max_replay_buffer_size, max_path_length, self._observation_dim)
        )
        self._actions = np.zeros(
            (max_replay_buffer_size, max_path_length, self._action_dim)
        )

        self._rewards = np.zeros(
            (max_replay_buffer_size, max_path_length, 1)
        )

        self._terminals = np.zeros(
            (max_replay_buffer_size, max_path_length, 1)
        )
        
        self._replace = replace
        self.batch_length = batch_length
        self.use_batch_length = use_batch_length
        
        self._top = 0
        self._size = 0

    def add_path(self, path):
        # TODO Account for multiple path lengths
        # path has the shape (path_length, n_env, dim)
        # buffer format has the shape(index, path_length, dim)
        # We discard the data at index 0
        new_data = slice(self._top, self._top + self.env.n_envs)
        self._observations[new_data] = path["observations"][1:].transpose(1, 0, 2)
        self._actions[new_data] = path["actions"][1:].transpose(1, 0, 2)

        # path['rewards'] has the shape (path_length, n_env)
        # buffer format has the shape (index, path_length, 1)
        self._rewards[new_data] = np.expand_dims(
            path["rewards"][1:].transpose(1, 0), -1
        )
        self._terminals[new_data] = np.expand_dims(
            path["terminals"][1:].transpose(1, 0), -1
        )

        self._advance()
    
    def _advance(self):
        # TODO: Find out when _top goes beyond replay buffer size, but size does not
        self._top = (self._top + self.env.n_envs) % self._max_replay_buffer_size
        
        if self._size < self._max_replay_buffer_size:
            self._size += self.env.n_envs
    
    def random_batch(self, batch_size):
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true \
                as the batch size is larger than the current size of the replay\
                buffer."
            )
    
        if self.use_batch_length:
            indices = np.random.choice(
                self._size, 
                size=batch_size,
                replace=True
            )

            # Starting positions for sampling trajectories
            batch_start = np.random.randint(
                0, self.max_path_length - self.batch_length, size=(batch_size)
            )

            # Get indices correponding to each member of the batch
            batch_indices = np.linspace(
                batch_start, 
                batch_start + self.batch_length,
                self.batch_length,
                endpoint=False
            ).astype(int)

            #TODO: Shouldn't this be [x,y,z]
            observations = self._observations[indices][
                np.arange(batch_size), batch_indices
            ]

            actions = self._actions[indices][
                np.arange(batch_size), batch_indices
            ]

            rewards = self._rewards[indices][
                np.arange(batch_size), batch_indices
            ]

            terminals = self._terminals[indices][
                np.arange(batch_size), batch_indices
            ]
            
        else:
            indices = np.random.choice(
                self._size, 
                size=batch_size,
                replace=self._replace or self._size < batch_size
            )

            observations = self._observations[indices]
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            terminals = self._terminals[indices]

        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )

        return batch

    def get_diagnostics(self):
        d = super().get_diagnostics()
        d["reward_in_buffer"] = self._rewards.sum()
        return d

