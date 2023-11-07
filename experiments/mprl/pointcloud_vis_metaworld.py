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
from rlkit.mprl.mp_env_metaworld import set_robot_based_on_ee_pos, get_object_pose_from_seg, get_geom_pose_from_seg, body_check_grasp
from rlkit.mprl.experiment import make_env
import open3d as o3d 

# flags i need to implement 
"""
1) use vision pose estimation
2) implement all hardcoded policies with vision
    - get all object geoms that are necessary, visualize and make sure they are working 
    - first try teleporting, then try mping to the appropriate position 
    - 
3) try setting up learned version 

geoms:
WrenchHandle for assembly and disassemble
"""

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


def get_depth_map(sim, camera_name, camera_height, camera_width):
    depth_map = get_camera_depth(
        sim=sim,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
    )
    depth_map = np.expand_dims(
        CU.get_real_depth_map(sim=sim, depth_map=depth_map), -1
    )
    plt.imshow(depth_map)
    plt.savefig(f"depth_map.png")
    plt.close()

def assembly():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="assembly-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    env.name="assembly-v2"
    o = env.reset()
    estimated_pos = get_geom_pose_from_seg(env, env.sim.model.geom_name2id("WrenchHandle"), 'corner', 640, 480, env.sim)
    set_robot_based_on_ee_pos(
        env,
        estimated_pos + np.array([0., 0., 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    for _ in range(20):
        env.step([0., 0., 0., 1.0])
        env.check_grasp(verify_stable_grasp=True)
    set_robot_based_on_ee_pos(
        env,
        env._eef_xpos + np.array([0., 0., 0.1]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        True,
    )
    frame = env.sim.render(camera_name="corner", width=500, height=500)
    plt.imshow(frame)
    plt.savefig("reset.png")
    

def disassemble():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="disassemble-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    o = env.reset()
    print(get_object_pose_from_seg(env, "WrenchHandle", "corner", 640, 480, env.sim))
    print(env._get_pos_objects())

def bin_picking():
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="stick-pull-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    o = env.reset()
    stick_pos = get_geom_pose_from_seg(
        env, 
        36,
        "corner2",
        500,
        500,
        env.sim        
    )
    env.name = "stick-pull-v2"
    set_robot_based_on_ee_pos(
        env, 
        stick_pos + np.array([-0.05, 0., 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    for _ in range(25):
        env.step(np.array([0., 0., 0., 1]))
    pail_pos = stick_pos = get_geom_pose_from_seg(
        env, 
        39,
        "corner2",
        500,
        500,
        env.sim        
    )
    set_robot_based_on_ee_pos(
        env, 
        pail_pos + np.array([-0.10, -0.05, -0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        True,
    )
    o, r, d, i = env.step(np.array([0., 0., 0., 1.0]))
    print(f"Info: {i}")
    seg = env.sim.render(camera_name="corner2", width=500, height=500)
    plt.imshow(seg)
    plt.savefig("seg.png")
    print(env.sim.model.geom_names)
    
def bin_picking_act():
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="bin-picking-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    seg = env.sim.render(camera_name="topview", width=500, height=500, segmentation=True)[:, :, 1]
    print(seg[350:355, 190:195])
    seg[350:355, 190:195] = 0
    plt.imshow(seg)
    plt.savefig("seg.png")
    print(env.sim.model.geom_names)
    obj_pose = get_geom_pose_from_seg(
        env, 
        36,
        "topview",
        480,
        640,
        env.sim        
    )
    print(f"Obj pose: {obj_pose}")
    print(f"actual puse: {env._get_pos_objects()}")

def hammer_act():
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="hammer-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    obj_pose = get_geom_pose_from_seg(
        env, 
        env.sim.model.geom_name2id("HammerHandle"),
        "topview",
        480,
        640,
        env.sim        
    )
    env.name = "hammer-v2"
    set_robot_based_on_ee_pos(
        env, 
        obj_pose + np.array([-0.0, 0., 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    seg = env.sim.render(camera_name="corner3", width=500, height=500)
    plt.imshow(seg)
    plt.savefig("seg.png")

def peg():
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="peg-insert-side-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    print(env.sim.model.geom_names)
    # visualize left and right pad geoms 
    from rlkit.mprl.mp_env_metaworld import geom_pointcloud
    leftpad_pcd = geom_pointcloud(
        env,
        env.sim.model.geom_name2id("leftpad_geom"),
        ["corner", "corner2"],
        500,
        500,
        env.sim,   
    )
    rightpad_pcd = geom_pointcloud(
        env,
        env.sim.model.geom_name2id("rightpad_geom"),
        ["corner", "corner2"],
        500,
        500,
        env.sim,   
    )
    object_pcd = geom_pointcloud(
        env,
        env.sim.model.geom_name2id("peg"),
        ["corner", "corner2"],
        500,
        500,
        env.sim,   
    )
    env.name = 'peg-insert-side-v2'
    o = env.reset()
    obj_pose = get_geom_pose_from_seg(
        env, 
        env.sim.model.geom_name2id("peg"),
        "topview",
        480,
        640,
        env.sim        
    )
    set_robot_based_on_ee_pos(
        env, 
        obj_pose + np.array([0.08, 0., 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    # do grasping 
    object_pcd = geom_pointcloud(
            env,
            env.sim.model.geom_name2id("peg"),
            ["corner", "corner2"],
            500,
            500,
            env.sim,   
        )
    from scipy.spatial.distance import cdist
    for _ in range(25):
        leftpad_pcd = geom_pointcloud(
            env,
            env.sim.model.geom_name2id("leftpad_geom"),
            ["corner", "corner2"],
            500,
            500,
            env.sim,   
        )
        rightpad_pcd = geom_pointcloud(
            env,
            env.sim.model.geom_name2id("rightpad_geom"),
            ["corner", "corner2"],
            500,
            500,
            env.sim,   
        )
        env.step(np.array([0., 0., 0., 1.0]))
        print(body_check_grasp(env), np.min(cdist(leftpad_pcd, object_pcd)), np.min(cdist(rightpad_pcd, object_pcd)))
    
    print(env.sim.model.geom_names)
    seg = env.sim.render(camera_name="corner2", width=500, height=500)
    plt.imshow(seg)
    plt.savefig("seg.png")

if __name__ == "__main__":
    assembly()