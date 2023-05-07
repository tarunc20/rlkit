import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import sys
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import rlkit.mopa.env
import cv2
from rlkit.mopa.env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.mopa.util.env import joint_convert, mat2quat, quat_mul, rotation_matrix, quat2mat
from rlkit.mopa.util.transform_utils import mat2pose, convert_quat, pose2mat
from rlkit.mopa.config import sawyer
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from rlkit.mopa.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
import mujoco_py
import collections 
import cv2

def mopa_test():
    config = ASSEMBLY_OBSTACLE_CONFIG 
    env = gym.make(**config)
    ik_env = gym.make(**config)
    env = MoPAMPEnv(
        "SawyerAssemblyObstacle-v0",
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
    # go down 
    for _ in range(15):
        ac = np.zeros(7)
        ac[:3] = np.array([0.,-0.01, -0.5])
        env.step(ac)
    save_img(env._wrapped_env, f"working.png")
    return 

if __name__ == "__main__":
    mopa_test()