import time

import cv2
import d4rl  # Import to register d4rl environments to Gym
import disrep4rl.environments.kitchen_custom_envs  # Import to register kitchen custom environments to Gym
import dm_env
import gym
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs

from rlkit.mprl.mp_env_kitchen import MPEnv

if __name__ == "__main__":
    np.random.seed(0)
    env_name = "kitchen-kettle-v0"
    env = gym.make(env_name)
    env = MPEnv(env)
    env.reset()
    t = time.time()
    num_steps = 1
    for step in range(num_steps):
        # o, r, d, i = env.step(np.concatenate((np.random.uniform(-1, 1, 6), [0])))
        cv2.imwrite(f"test_{step}.png", env.get_image())
        # if d:
        #     env.reset()
    print("FPS: ", num_steps / (time.time() - t))
