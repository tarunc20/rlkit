import random
import time
import mujoco_py
import cv2
import numpy as np
import robosuite
import robosuite.utils.camera_utils as CU
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from urdfpy import URDF
import collections
import matplotlib.pyplot as plt

from rlkit.mprl.mp_env import (
    RobosuiteEnv,
    check_robot_collision,
    compute_pcd,
    pcd_collision_check,
    get_camera_depth 
)
from rlkit.mprl.experiment import make_env
import open3d as o3d 
from rlkit.mopa.mopa_env import MoPAMPEnv, get_site_pose, set_robot_based_on_ee_pos, get_object_pose, cart2joint_ac
import gym
from mopa_rl.config.default_configs import LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG, LIFT_CONFIG, PUSHER_OBSTACLE_CONFIG

def get_object_pose_from_seg(env, object_string, camera_name, camera_width, camera_height, sim):
    segmentation_map = CU.get_camera_segmentation(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    obj_id = sim.model.geom_name2id(object_string)
    obj_mask = segmentation_map == obj_id 
    depth_map = get_camera_depth(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    depth_map = np.expand_dims(
        CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
    )
    world_to_camera = CU.get_camera_transform_matrix(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    obj_pointcloud = CU.transform_from_pixels_to_world(
        pixels=np.argwhere(obj_mask),
        depth_map=depth_map[..., 0],
        camera_to_world_transform=camera_to_world,
    )
    return np.mean(obj_pointcloud, axis = 0)

def save_img(env, filename):
    frame = env.sim.render(height=500, width=500,camera_name="eye_in_hand")
    plt.imshow(frame)
    plt.savefig(filename)
    plt.close()

def test_lift():
    np.random.seed(0)
    LIFT_OBSTACLE_CONFIG["camera_name"] = "eye_in_hand"
    env = gym.make(**LIFT_CONFIG)
    ik_env = gym.make(**LIFT_CONFIG)
    o = env.reset()
    # get list of allowed collision pairs 
    obj_pos = get_object_pose_from_seg(
        env=env, 
        object_string="cube", 
        camera_name="topview", 
        camera_width=640, 
        camera_height=480, 
        sim=env.sim)
    desired_xpos = obj_pos + np.array([0., 0., 0.02])
    quat = np.array([-0.1268922, 0.21528646, 0.96422245, -0.08846001])
    quat /= np.linalg.norm(quat)

    gripper_pos = get_site_pose(env, "grip_site")[0]
    ac = collections.OrderedDict()
    ac['default'] = gripper_pos #obj_pos - gripper_pos 
    ac['quat'] = np.array([1., 0., 0., 0.]) #quat
    allowed_collision_pairs = []
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
    result, err_norm = set_robot_based_on_ee_pos(
        env,
        ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        False,
        "cube",
        LIFT_OBSTACLE_CONFIG,
        allowed_collision_pairs,
        "SawyerLiftObstacle-v0",
    )
    print(f"Result: {result}")
    print(f"Err norm: {err_norm}")
    save_img(env, "test.png")

def test_assembly():
    ASSEMBLY_OBSTACLE_CONFIG["screen_height"] = 500
    ASSEMBLY_OBSTACLE_CONFIG["screen_width"] = 500
    ASSEMBLY_OBSTACLE_CONFIG["camera_name"] = "eye_in_hand"
    env = gym.make(**ASSEMBLY_OBSTACLE_CONFIG)
    ik_env = gym.make(**ASSEMBLY_OBSTACLE_CONFIG)
    o = env.reset()
    print(env.sim.model.geom_names)
    hole_pos = get_site_pose(env, "hole")[0]
    print(f"Position of hole: {hole_pos}")
    allowed_collision_pairs = []
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in env.static_geom_ids:
            allowed_collision_pairs.append(
                (manipulation_geom_id, geom_id)
            )
    # save segmentation
    # segmentation_map = CU.get_camera_segmentation(
    #     camera_name="birdview",
    #     camera_width=640,
    #     camera_height=480,
    #     sim=env.sim,
    # )
    # plt.imshow(segmentation_map[:, :, 1])
    # plt.savefig("seg.png")
    #for name in env.sim.model.geom_names:
    pos = get_object_pose_from_seg(env, "4_part4_mesh", "topview", 640, 480, env.sim)
    quat = np.array([-0.69904332, -0.35891423, -0.60671187,  0.12008213])
    ac = collections.OrderedDict()
    ac["default"] = pos + np.array([0.0, -0.3, 0.4]) #np.array([0.15, 0.10, 0.3])
    ac["quat"] = quat
    result, err_norm = set_robot_based_on_ee_pos(
            env,
            ac,
            ik_env,
            env.sim.data.qpos.copy(),
            env.sim.data.qvel.copy(), 
            False,
            "cube",
            ASSEMBLY_OBSTACLE_CONFIG,
            allowed_collision_pairs,
            "SawyerAssemblyObstacle-v0"
        )
    print(f"result: {result}")
    print(f"err norm: {err_norm}")
    # go towards site 
    # grip_site = get_site_pose(env, "grip_site")[0]
    # ac = collections.OrderedDict()
    # ac["default"] = hole_pos - grip_site + np.array([0., 0., 0.3])
    # converted_ac = cart2joint_ac(
    #         env, 
    #         ik_env,
    #         ac,
    #         env.sim.data.qpos.copy(),
    #         env.sim.data.qvel.copy(),
    #         ASSEMBLY_OBSTACLE_CONFIG,
    #     )
    # for _ in range(25):
    #     env.step(converted_ac)
    save_img(env, "attempt.png")

def test_push():
    PUSHER_OBSTACLE_CONFIG["camera_name"] = "eye_in_hand"
    env = gym.make(**PUSHER_OBSTACLE_CONFIG)
    ik_env = gym.make(**PUSHER_OBSTACLE_CONFIG)
    o = env.reset()
    cube_pos = get_site_pose(env, "cube")[0]
    print(f"Cube pos : {cube_pos}")
    allowed_collision_pairs = []
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in env.static_geom_ids:
            allowed_collision_pairs.append(
                (manipulation_geom_id, geom_id)
            )
    est_cube_pos = get_object_pose_from_seg(
        env, "cube", "frontview", 640, 480, env.sim)
    ac = collections.OrderedDict()
    ac["default"] = est_cube_pos + np.array([-0.1, 0.04, 0.06])
    result, err_norm = set_robot_based_on_ee_pos(
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
    save_img(env, "attempt.png")
    
def test_all_envs():
    PUSHER_OBSTACLE_CONFIG["screen_height"] = 500
    PUSHER_OBSTACLE_CONFIG["screen_width"] = 500
    PUSHER_OBSTACLE_CONFIG["camera_name"] = "eye_in_hand"
    env = gym.make(**PUSHER_OBSTACLE_CONFIG)
    ik_env = gym.make(**PUSHER_OBSTACLE_CONFIG)
    env = MoPAMPEnv(
        "SawyerPushObstacle-v0",
        env,
        ik_env, 
        planner_command_orientation=True,
        teleport_on_grasp=True,
        use_vision_pose_estimation=True,
        config=PUSHER_OBSTACLE_CONFIG,
    )
    o = env.reset()
    save_img(env._wrapped_env, "attempt.png")
if __name__ == "__main__":
    test_push()