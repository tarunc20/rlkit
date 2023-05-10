import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import mujoco_py
import sys
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from mopa_rl.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
import collections 
import cv2
from robosuite.utils.transform_utils import mat2euler, euler2mat

def mopa_test():
    #np.random.seed(0)
    config = LIFT_CONFIG 
    config["camera"]='visview'
    env = gym.make(**config)
    ik_env = gym.make(**config)
    env = MoPAMPEnv(
        "SawyerLift-v0",
        env,
        ik_env,
        config=config,
        plan_to_learned_goals=False,
        num_ll_actions_per_hl_action=25,
        teleport_on_grasp=False,
        vertical_displacement=0.05,
        planner_command_orientation=True,
    )
    o = env.reset()
    save_img(env._wrapped_env, f"start.png")
    ac = np.zeros(7)
    ac[:3] = np.array([0., 0., -1.0])
    env.step(ac)
    ac = np.zeros(7)
    ac[-1] = 0.5
    env.step(ac)
    ac = np.zeros(7)
    ac[:3] = np.array([0., 0., 0.9])
    ac[-1] = 2.0
    for _ in range(40):
        env.step(ac)
    print(f"Checking success: {env._wrapped_env._success}")
    save_img(env._wrapped_env, f"working.png")
    return 

if __name__ == "__main__":
    mopa_test()