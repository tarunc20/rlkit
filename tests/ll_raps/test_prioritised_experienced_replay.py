import numpy as np
import pdb
from gym.envs.mujoco import HalfCheetahEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBufferLowLevelRAPS,
    )

def test_prioritised_replay():
    env = NormalizedBoxEnv(HalfCheetahEnv())
    buffer = EpisodeReplayBufferLowLevelRAPS(
        env, 
        20, 
        5, 
        1000, 
        100, 
        5, 
        5, 
        True, 
        50, 
        False,
        1,
        False
    )

    assert buffer is not None

    buffer._rewards = np.zeros((10, 100, 1))
    buffer._size = 10
    for i in range(10):
        buffer._rewards[i, -1, -1] = (i + 1) * 100
    
    counts = [0] * 10
    iters = 10000
    eps = 1e-4
    
    for i in range(iters):
        index = int(buffer.random_batch(1)['rewards'].sum()) // 100
        counts[index-1] += 1

    for index, count in enumerate(counts):
        assert count/iters - (index + 1)/10 <= eps
