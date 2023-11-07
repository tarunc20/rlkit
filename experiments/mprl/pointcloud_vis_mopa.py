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
from rlkit.mopa.mopa_env import MoPAMPEnv
import gym
from mopa_rl.config.default_configs import LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mprl.mp_env import get_camera_depth 
def main():
    np.random.seed(0)
    ASSEMBLY_OBSTACLE_CONFIG["camera_name"]="eye_in_hand"
    camera_name = 'eye_in_hand'
    env = gym.make(**ASSEMBLY_OBSTACLE_CONFIG)
    o = env.reset()
    
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig("frame.png")
    assert False
    sim = env.sim
    segmentation_map = CU.get_camera_segmentation(
        camera_name=camera_name,
        camera_width=640,
        camera_height=480,
        sim=sim,
    )
    plt.imshow(segmentation_map[:, :, 1])
    plt.savefig("seg.png")
    depth_map = get_camera_depth(
        camera_name=camera_name,
        camera_width=640,
        camera_height=480,
        sim=sim,
    )
    depth_map = np.expand_dims(
        CU.get_real_depth_map(sim=sim, depth_map=depth_map), -1
    )
    obj_id = sim.model.geom_name2id("cube")
    obj_mask = segmentation_map[:, :, 1] == obj_id 
    plt.imshow(1. - obj_mask)
    world_to_camera = CU.get_camera_transform_matrix(
        camera_name=camera_name,
        camera_width=640,
        camera_height=480,
        sim=sim,
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    all_img_pixels = np.argwhere(
            segmentation_map[:, :, 1] + 5,
    )
    pointcloud = CU.transform_from_pixels_to_world(
        pixels=all_img_pixels,
        depth_map=depth_map[..., 0],
        camera_to_world_transform=camera_to_world,
    )
    obj_pointcloud = CU.transform_from_pixels_to_world(
        pixels=np.argwhere(obj_mask),
        depth_map=depth_map[..., 0],
        camera_to_world_transform=camera_to_world,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud("test_pcd.pcd", pcd)
    print(f"Estimated position: {np.mean(obj_pointcloud, axis = 0)}")
    print(f"True position: {env.sim.data.body_xpos[env.sim.model.body_name2id('cube')]}")

if __name__ == "__main__":
    main()

