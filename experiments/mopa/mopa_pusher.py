import numpy as np
import mujoco_py
import gym
from tqdm import tqdm
from gym import spaces
import sys
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from mopa_rl.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG, PUSHER_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
import collections 
import cv2

def print_all_collisions(env):
    mjcontacts = env.sim.data.contact
    ncon = env.sim.data.ncon
    for i in range(ncon):
        ct = mjcontacts[i]
        ct1 = ct.geom1; ct2 = ct.geom2 
        b1 = env.sim.model.geom_bodyid[ct1]
        b2 = env.sim.model.geom_bodyid[ct2]
        bn1 = env.sim.model.body_id2name(b1)
        bn2 = env.sim.model.body_id2name(b2)
        print(f"Body name 1: {bn1}")
        print(f"Body name 2: {bn2}")

def main():
    env = gym.make(**PUSHER_OBSTACLE_CONFIG)
    ik_env = gym.make(**PUSHER_OBSTACLE_CONFIG)
    o = env.reset()
    print("Initial collisions")
    print_all_collisions(env)
    # create list of allowed collision pairs 
    allowed_collision_pairs = []
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in env.static_geom_ids:
            allowed_collision_pairs.append(
                (manipulation_geom_id, geom_id)
            )
    # get cube position 
    right_gripper, left_gripper = (
            env.sim.data.get_site_xpos("right_eef"),
            env.sim.data.get_site_xpos("left_eef"),
        )
    gripper_site_pos = (right_gripper + left_gripper) / 2.0
    cube_pos = np.array(env.sim.data.body_xpos[env.cube_body_id])
    target_ac = cube_pos - gripper_site_pos + np.array([-0.05, 0.05, 0.06])
    ac = OrderedDict()
    ac['default'] = target_ac 
    set_robot_based_on_ee_pos(
        env,
        ac,
        ik_env, 
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
        "cube",
        PUSHER_OBSTACLE_CONFIG,
        allowed_collision_pairs,
        "SawyerPushObstacle-v0"
    )
    print("Collisions at end")
    print_all_collisions(env)
    save_img(env, "new.png")
    # keep pushing towards object
    right_gripper, left_gripper = (
        env.sim.data.get_site_xpos("right_eef"),
        env.sim.data.get_site_xpos("left_eef"),
    )
    gripper_site_pos = (right_gripper + left_gripper) / 2.0
    cube_pos = np.array(env.sim.data.body_xpos[env.cube_body_id])
    # ac = OrderedDict()
    # ac['default'] = cube_pos - gripper_site_pos 
    # converted_ac = cart2joint_ac(
    #     env,
    #     ik_env,
    #     ac,
    #     env.sim.data.qpos.copy(),
    #     env.sim.data.qvel.copy(),
    #     PUSHER_OBSTACLE_CONFIG,
    # )
    # for _ in range(5):
    #     o, r, d, i = env.step(converted_ac)
    ac = OrderedDict()
    ac['default'] = np.array([1.0, 0.0, 0.01])
    converted_ac = cart2joint_ac(
        env,
        ik_env,
        ac,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        PUSHER_OBSTACLE_CONFIG,
    )
    for _ in range(1):
        o, r, d, i = env.step(converted_ac)
    ac = OrderedDict()
    ac['default'] = np.array([0.0, 0.0, -0.03])
    converted_ac = cart2joint_ac(
        env,
        ik_env,
        ac,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        PUSHER_OBSTACLE_CONFIG,
    )
    for _ in range(1):
        o, r, d, i = env.step(converted_ac)
    save_img(env, "end.png")

if __name__ == "__main__":
    main()