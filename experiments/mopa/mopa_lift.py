import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import mujoco_py
import sys
from PIL import Image
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from mopa_rl.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
import collections 
import cv2
from robosuite.utils.transform_utils import mat2euler, euler2mat, convert_quat

def get_camera_segmentation(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera segmentation matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): 2-channel segmented image where the first contains the
            geom types and the second contains the geom IDs
    """
    return sim.render(camera_name=camera_name, height=camera_height, width=camera_width, segmentation=True)[::-1]

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
    #np.random.seed(0)
    config = LIFT_OBSTACLE_CONFIG 
    config["camera_name"]='visview'
    env = gym.make(**config)
    ik_env = gym.make(**config)
    env = MoPAMPEnv(
        "SawyerLiftObstacle-v0",
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
    ac = np.zeros(7)
    ac[:3] = np.array([0., 0., -1.0])
    ac[-1] = -1.0
    for _ in range(3):
        env.step(ac)
    ac = np.zeros(7)
    ac[-1] = 1.0
    for _ in range(2):
        env.step(ac)
    ac = np.zeros(7)
    ac[:3] = np.array([0., 0., 1.0])
    ac[-1] = 1.0
    for _ in range(15):
        env.step(ac)
    # for _ in range(2):
    #     env.step(ac)
    # save_img(env._wrapped_env, f"start.png")
    # ac = np.zeros(7)
    # ac[-1] = 5.0
    # env.step(ac)
    # ac = np.zeros(7)
    # ac[:3] = np.array([0., 0., 0.9])
    # ac[-1] = 2.0
    # for _ in range(20):
    #     env.step(ac)
    # print(f"Checking success: {env._wrapped_env._success}")
    save_img(env._wrapped_env, f"working.png")
    return 


def test_mopa_new_orientation():
    config = LIFT_CONFIG 
    config["camera_name"]='agentview'
    print(f"Config: {config}")
    config["screen_width"] = 500
    config["screen_height"] = 500
    allowed_collision_pairs = []
    env = gym.make(**config)
    ik_env = gym.make(**config)
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in env.static_geom_ids:
            allowed_collision_pairs.append(
                (manipulation_geom_id, geom_id)
            )
    for manipulation_geom_id in env.manipulation_geom_ids:
        for lf in env.left_finger_geoms:
            allowed_collision_pairs.append(
                (manipulation_geom_id, env.sim.model.geom_name2id(lf))
            )
    for manipulation_geom_id in env.manipulation_geom_ids:
        for rf in env.right_finger_geoms:
            allowed_collision_pairs.append(
                (manipulation_geom_id, env.sim.model.geom_name2id(rf))
            )
    result = False
    i = 0
    while not result:
        o = env.reset()
        ac = OrderedDict()
        ac["default"] = get_object_pose(env, "cube")[:3] + np.array([0., 0.0, 0.07]) - get_site_pose(env, "grip_site")[0]
        quat = np.array([-0.1268922, 0.21528646, 0.96422245, -0.08846001])
        quat /= np.linalg.norm(quat)
        #quat = np.array([1., 0., 0., 0.])
        print(f"Try: {i} quat: {quat}")
        i += 1
        ac["quat"] = quat
        result, err_norm = set_robot_based_on_ee_pos(
            env,
            ac,
            ik_env,
            env.sim.data.qpos.copy(),
            env.sim.data.qvel.copy(), 
            False,
            "cube",
            config,
            allowed_collision_pairs,
            "SawyerLift-v0"
        )
        ac["default"] = np.array([0., 0., -1.0])
        ac["gripper"] = [-1.0]
        converted_ac = cart2joint_ac(
            env, 
            ik_env,
            ac,
            env.sim.data.qpos.copy(),
            env.sim.data.qvel.copy(),
            LIFT_CONFIG,
        )
        for _ in range(3):
            env.step(converted_ac)
        ac["default"] = np.array([0., 0., 0.0])
        ac["gripper"] = [1.0]
        converted_ac = cart2joint_ac(
            env, 
            ik_env,
            ac,
            env.sim.data.qpos.copy(),
            env.sim.data.qvel.copy(),
            LIFT_CONFIG,
        )
        for _ in range(2):
            env.step(converted_ac)
        ac["default"] = np.array([0., 0., 1.0])
        ac["gripper"] = [1.0]
        converted_ac = cart2joint_ac(
            env, 
            ik_env,
            ac,
            env.sim.data.qpos.copy(),
            env.sim.data.qvel.copy(),
            LIFT_CONFIG,
        )
        for _ in range(10):
            env.step(converted_ac)
        save_img(env, f"try_{i}.png")

def mp_test():
    np.random.seed(0)
    config = LIFT_CONFIG 
    config["camera_name"]='agentview'
    config["screen_width"]=500
    config["screen_height"]=500
    env = gym.make(**config)
    ik_env = gym.make(**config)
    # set up allowed collision pairs 
    allowed_collision_pairs= []
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in env.static_geom_ids:
            allowed_collision_pairs.append(
                (manipulation_geom_id, geom_id)
            )
    for manipulation_geom_id in env.manipulation_geom_ids:
        for lf in env.left_finger_geoms:
            allowed_collision_pairs.append(
                (manipulation_geom_id, env.sim.model.geom_name2id(lf))
            )
    for manipulation_geom_id in env.manipulation_geom_ids:
        for rf in env.right_finger_geoms:
            allowed_collision_pairs.append(
                (manipulation_geom_id, env.sim.model.geom_name2id(rf))
            )
    env.reset()
    save_img(env, "reset.png")
    ac = collections.OrderedDict()
    # ac["default"] = get_site_pose(env, "grip_site")[0].astype(np.float64)
    # ac["quat"] = convert_quat(get_site_pose(env, "grip_site")[1], to="wxyz")
    cube_pos = get_object_pose(env, "cube")[:3].copy() + np.array([0., 0.00, 0.05])
    print(f"Cube pos: {cube_pos}")
    gripper_pos = get_site_pose(env, "grip_site")[0]
    print(f"Original gripper pos: {gripper_pos, get_site_pose(env, 'grip_site')[1]}")
    ac["default"] = cube_pos# - gripper_pos
    quat = np.array([-0.1268922, 0.21528646, 0.96422245, -0.08846001])
    quat /= np.linalg.norm(quat)
    ac["quat"] = quat
    # # for mp to point, pass quat in as wxyz
    mp_to_point(
        env,
        ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        allowed_collision_pairs,
        is_grasped=False,
        ignore_object_collision=False,
    )
    save_img(env, "set.png")

if __name__ == "__main__":
    mp_test()