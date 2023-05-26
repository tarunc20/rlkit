import numpy as np
import mujoco_py
import gym
from tqdm import tqdm
from PIL import Image
from gym import spaces
import sys
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from mopa_rl.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
import collections 
import cv2
def save_img(env, filename):
    frame = (env.render("rgb_array")*255.0).astype(np.uint8)
    # plt.imshow(frame)
    # plt.savefig(filename)
    # plt.close()
    img = Image.fromarray(frame, "RGB")
    img.show()
    img.save(filename)
    return 

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
    save_img(env._wrapped_env, f"start.png")
    # go down 
    for _ in range(15):
        ac = np.zeros(7)
        ac[:3] = np.array([0.,-0.01, -0.5])
        env.step(ac)
    #save_img(env._wrapped_env, f"working.png")
    return 

if __name__ == "__main__":
    mopa_test()