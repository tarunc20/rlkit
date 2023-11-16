import copy
import io
import xml.etree.ElementTree as ET

import cv2
import gym
import numpy as np
import robosuite.utils.transform_utils as T
from gym import spaces
from robosuite.controllers import controller_factory
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper

from rlkit.core import logger
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.mprl import module
from rlkit.mprl.inverse_kinematics import qpos_from_site_pose_kitchen
from rlkit.torch.model_based.dreamer.visualization import add_text

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join
    try:
        sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
        from ompl import base as ob
        from ompl import geometric as og
        from ompl import util as ou
    except:
        pass 


def get_object_string(env, obj_idx=0):
    element = env.TASK_ELEMENTS[obj_idx]
    return element


################## VISION PIPELINE ##################
def get_camera_depth(sim, camera_name, camera_height, camera_width):
    """
    Obtains depth image.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): the depth image b/w 0 and 1
    """
    return sim.render(
        camera_name=camera_name, height=camera_height, width=camera_width, depth=True
    )[1][::-1]


def compute_object_pcd(
    env,
    camera_height=480,
    camera_width=640,
    grasp_pose=True,
    target_obj=False,
    obj_idx=0,
):
    name = env.name.split("_")[1]
    object_pts = []
    if target_obj:
        camera_names = ["agentview", "birdview"]
        # need birdview to properly estimate bin position
    else:
        camera_names = ["agentview", "sideview"]
    for camera_name in camera_names:
        sim = env.sim
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        depth_map = get_camera_depth(
            sim=sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
        )

        # get camera matrices
        world_to_camera = CU.get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        camera_to_world = np.linalg.inv(world_to_camera)

        # get robot segmentation mask
        geom_ids = np.unique(segmentation_map[:, :, 1])
        object_ids = []
        if grasp_pose or not target_obj:
            object_string = get_object_string(env, obj_idx=obj_idx)
            if "Door" in name:
                object_string = "handle"
        else:
            if "NutAssembly" in name:
                if name.endswith("Square"):
                    object_string = "peg1"
                elif name.endswith("Round"):
                    object_string = "peg2"
                else:
                    if obj_idx == 0:
                        object_string = "peg2"
                    else:
                        object_string = "peg1"
            if "PickPlace" in name:
                object_string = "full_bin"
        for i, geom_id in enumerate(geom_ids):
            geom_name = sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if object_string in geom_name:
                if "NutAssembly" in name and grasp_pose:
                    if name.endswith("Square"):
                        target_geom_id = "g4_visual"
                    elif name.endswith("Round"):
                        target_geom_id = "g8_visual"
                    elif name.endswith("NutAssembly"):
                        if obj_idx == 0:
                            target_geom_id = "g8_visual"
                        else:
                            target_geom_id = "g4_visual"
                    if geom_name.endswith(target_geom_id):
                        object_ids.append(geom_id)
                else:
                    object_ids.append(geom_id)
        if len(object_ids) > 0:
            if target_obj and "PickPlace" in name:
                full_bin_mask = segmentation_map[:, :, 1] == object_ids[0]
                clust_img, clust_masks = pcv.spatial_clustering(
                    full_bin_mask.astype(np.uint8) * 255,
                    algorithm="DBSCAN",
                    min_cluster_size=5,
                    max_distance=None,
                )
                new_obj_idx = compute_correct_obj_idx(env, obj_idx=obj_idx)
                clust_masks = [clust_masks[i] for i in [0, 2, 1, 3]]
                object_mask = clust_masks[new_obj_idx]
            else:
                object_mask = np.any(
                    [
                        segmentation_map[:, :, 1] == object_id
                        for object_id in object_ids
                    ],
                    axis=0,
                )
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            object_pts.append(object_pointcloud)

    # if object_pts is empty, return the value from the last time this function was called with the same args
    # this is a bit of a hack, but necessary since the object may be occluded sometimes
    if len(object_pts) > 0:
        env.cache[
            (camera_height, camera_width, grasp_pose, target_obj, obj_idx)
        ] = object_pts
    else:
        object_pts = env.cache[
            (camera_height, camera_width, grasp_pose, target_obj, obj_idx)
        ]
    object_pointcloud = np.concatenate(object_pts, axis=0)
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_pointcloud)
    cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    object_pcd = object_pcd.select_by_index(ind)
    object_xyz = np.array(object_pcd.points)
    return object_xyz


def compute_pcd(
    env,
    obj_idx=0,
    camera_height=480,
    camera_width=640,
    is_grasped=False,
):
    pts = []
    object_pts = []
    camera_names = ["agentview", "birdview"]
    for camera_name in camera_names:
        sim = env.sim
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        depth_map = get_camera_depth(
            sim=sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
        )

        # get camera matrices
        world_to_camera = CU.get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        camera_to_world = np.linalg.inv(world_to_camera)

        # get robot segmentation mask
        geom_ids = np.unique(segmentation_map[:, :, 1])
        robot_ids = []
        object_ids = []
        object_string = get_object_string(env, obj_idx=obj_idx)
        for geom_id in geom_ids:
            geom_name = sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if geom_name.startswith("robot0") or geom_name.startswith("gripper"):
                robot_ids.append(geom_id)
            if object_string in geom_name:
                object_ids.append(geom_id)
        robot_mask = np.any(
            [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids], axis=0
        )
        if is_grasped and len(object_ids) > 0:
            object_mask = np.any(
                [segmentation_map[:, :, 1] == object_id for object_id in object_ids],
                axis=0,
            )
            cv2.imwrite(
                f"object_mask_{camera_name}.png", object_mask.astype(np.uint8) * 255
            )
            # only remove object from scene if it is grasped
            all_img_pixels = np.argwhere(
                1 - robot_mask - object_mask
            )  # remove robot from scene pcd
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            object_pts.append(object_pointcloud)
        else:
            all_img_pixels = np.argwhere(1 - robot_mask)
        # transform from camera pixel back to world position
        pointcloud = CU.transform_from_pixels_to_world(
            pixels=all_img_pixels,
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        pts.append(pointcloud)

    pointcloud = np.concatenate(pts, axis=0)
    pointcloud = pointcloud[pointcloud[:, -1] > 0.75]
    pointcloud = pointcloud[pointcloud[:, -1] < 1.0]
    pointcloud = pointcloud[pointcloud[:, 0] > -0.3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    xyz = np.array(pcd.points)
    if is_grasped and len(object_pts) > 0:
        object_pointcloud = np.concatenate(object_pts, axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pointcloud)
        cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        object_pcd = object_pcd.select_by_index(ind)
        object_xyz = np.array(object_pcd.points)
    else:
        object_xyz = None

    return xyz, object_xyz


def pcd_collision_check(
    env,
    target_angles,
    gripper_qpos,
    is_grasped,
):
    xyz, object_pts = env.xyz, env.object_pcd
    robot = env.robot
    joints = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    combined = []
    qpos = np.concatenate([target_angles, gripper_qpos])
    base_xpos = env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_link0")]
    fk = robot.collision_trimesh_fk(dict(zip(joints, qpos)))
    link_fk = robot.link_fk(dict(zip(joints, qpos)))
    mesh_base_xpos = link_fk[robot.links[0]][:3, 3]
    for mesh, pose in fk.items():
        pose[:3, 3] = pose[:3, 3] + (base_xpos - mesh_base_xpos)
        homogenous_vertices = np.concatenate(
            [mesh.vertices, np.ones((mesh.vertices.shape[0], 1))], axis=1
        ).astype(np.float32)
        transformed = np.matmul(pose.astype(np.float32), homogenous_vertices.T).T[:, :3]
        mesh_new = trimesh.Trimesh(transformed, mesh.faces)
        combined.append(mesh_new)

    combined_mesh = trimesh.util.concatenate(combined)
    robot_mesh = combined_mesh.as_open3d
    # transform object pcd by amount rotated/moved by eef link
    # compute the transform between the old and new eef poses

    # note: this is just to get the forward kinematics using the sim,
    # faster/easier that way than using trimesh fk
    # implementation detail, not important
    old_eef_xquat = env._eef_xquat.copy()
    old_eef_xpos = env._eef_xpos.copy()
    old_qpos = env.sim.data.qpos.copy()

    env.robots[0].set_robot_joint_positions(target_angles)

    ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
    ee_new_mat = pose2mat((env._eef_xpos, env._eef_xquat))
    transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

    env.robots[0].set_robot_joint_positions(old_qpos[:7])

    # Create a scene and add the triangle mesh

    if is_grasped:
        object_pts = object_pts @ transform[:3, :3].T + transform[:3, 3]
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pts)
        hull, _ = object_pcd.compute_convex_hull()
        # compute pcd distance to xyz
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(hull + robot_mesh)
        )  # we do not need the geometry ID for mesh
        occupancy = scene.compute_occupancy(xyz.astype(np.float32), nthreads=32)
        collision = sum(occupancy.numpy()) > 5
    else:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(robot_mesh)
        )  # we do not need the geometry ID for mesh
        occupancy = scene.compute_occupancy(xyz.astype(np.float32), nthreads=32)
        collision = sum(occupancy.numpy()) > 5
    return collision


def grasp_pcd_collision_check(
    env,
    obj_idx=0,
):
    xyz = compute_object_pcd(
        env,
        obj_idx=obj_idx,
        grasp_pose=False,
        target_obj=False,
        camera_height=256,
        camera_width=256,
    )
    robot = env.robot
    joints = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    combined = []

    # compute floating gripper mesh at the correct pose
    base_xpos = env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_link0")]
    link_fk = robot.link_fk(dict(zip(joints, env.sim.data.qpos[:9])))
    mesh_base_xpos = link_fk[robot.links[0]][:3, 3]
    combined = []
    for link in robot.links[-2:]:
        pose = link_fk[link]
        pose[:3, 3] = pose[:3, 3] + (base_xpos - mesh_base_xpos)
        homogenous_vertices = np.concatenate(
            [
                link.collision_mesh.vertices,
                np.ones((link.collision_mesh.vertices.shape[0], 1)),
            ],
            axis=1,
        )
        transformed = np.matmul(pose, homogenous_vertices.T).T[:, :3]
        mesh_new = trimesh.Trimesh(transformed, link.collision_mesh.faces)
        combined.append(mesh_new)
    robot_mesh = trimesh.util.concatenate(combined).as_open3d

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(robot_mesh)
    )  # we do not need the geometry ID for mesh
    sdf = scene.compute_signed_distance(xyz.astype(np.float32), nthreads=32).numpy()
    collision = np.any(sdf < 0.001)
    return collision


################## VISION PIPELINE ##################


def get_object_pose_mp(env, obj_idx=0):
    """
    Note this is only used for computing the target for MP
    this is NOT the true object pose
    """
    element = env.TASK_ELEMENTS[obj_idx]
    if element == "slide cabinet":
        object_pos = env.get_site_xpos("schandle1")
        object_quat = np.zeros(4)  # doesn't really matter
    elif element == "top burner":
        object_pos = env.get_site_xpos("tlbhandle")
        object_quat = np.zeros(4)  # doesn't really matter
    elif element == "hinge cabinet":
        object_pos = env.get_site_xpos("hchandle1")
        object_quat = np.zeros(4)  # doesn't really matter
    elif element == "light switch":
        object_pos = env.get_site_xpos("lshandle1")
        object_quat = np.zeros(4)  # doesn't really matter
    elif element == "microwave":
        object_pos = env.get_site_xpos("mchandle1")
        object_quat = np.zeros(4)  # doesn't really matter
    elif element == "kettle":
        object_pos = env.get_site_xpos("khandle1")
        object_quat = np.zeros(4)  # doesn't really matter
    return object_pos, object_quat


def get_object_pose(env, obj_idx=0):
    element = env.TASK_ELEMENTS[obj_idx]
    object_qpos = env.sim.data.qpos[-21:]
    object_pos = object_qpos[env.OBS_ELEMENT_INDICES[element] - 9]
    object_quat = np.zeros(4)  # doesn't really matter
    return object_pos, object_quat


def set_object_pose(env, object_pos, object_quat, obj_idx=0):
    """
    Set the object pose in the environment.
    Makes sure to convert from xyzw to wxyz format for quaternion. qpos requires wxyz!
    Arguments:
        env
        object_pos (np.ndarray): 3D position of the object
        object_quat (np.ndarray): 4D quaternion of the object (xyzw format)

    """
    element = env.TASK_ELEMENTS[obj_idx]
    env.sim.data.qpos[-21 + env.OBS_ELEMENT_INDICES[element] - 9] = object_pos
    env.sim.forward()


def check_object_grasp(env, obj_idx=0):
    element = env.TASK_ELEMENTS[obj_idx]
    is_grasped = False

    if element == "slide cabinet":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("schandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.07
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.07
            right = right_pad[0] < obj_pos[0]
            left = obj_pos[0] < left_pad[0]
            if right and left and within_sphere_right and within_sphere_left:
                is_grasped = True
    if element == "top burner":
        obj_pos = env.get_site_xpos("tlbhandle")
        left_pad = env.get_site_xpos("leftpad")
        right_pad = env.get_site_xpos("rightpad")
        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.035
        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.04
        right = right_pad[0] < obj_pos[0]
        left = obj_pos[0] < left_pad[0]
        if within_sphere_right and within_sphere_left and right and left:
            is_grasped = True
    if element == "microwave":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("mchandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
            if (
                right_pad[0] < obj_pos[0]
                and obj_pos[0] < left_pad[0]
                and within_sphere_right
                and within_sphere_left
            ):
                is_grasped = True
    if element == "hinge cabinet":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("hchandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.06
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.06
            if (
                right_pad[0] < obj_pos[0]
                and obj_pos[0] < left_pad[0]
                and within_sphere_right
            ):
                is_grasped = True
    if element == "light switch":
        for i in range(1, 4):
            obj_pos = env.get_site_xpos("lshandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.045
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.03
            if within_sphere_right and within_sphere_left:
                is_grasped = True
    if element == "kettle":
        # TODO: check if kettle is grasped
        pass
    return is_grasped


def set_robot_based_on_ee_pos(
    env,
    target_pos,
    target_quat,
    qpos,
    qvel,
    is_grasped,
    obj_idx=0,
    open_gripper_on_tp=False,
):
    """
    Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
    If grasping an object, ensures the object moves with the arm in a consistent way.
    """
    # cache quantities from prior to setting the state
    object_pos, object_quat = get_object_pose(env, obj_idx=obj_idx)
    object_pos = object_pos.copy()
    object_quat = object_quat.copy()
    gripper_qpos = env.sim.data.qpos[7:9].copy()
    gripper_qvel = env.sim.data.qvel[7:9].copy()
    old_eef_xquat = env._eef_xquat.copy()
    old_eef_xpos = env._eef_xpos.copy()

    # reset to canonical state before doing IK
    env.sim.data.qpos[:7] = qpos[:7]
    env.sim.data.qvel[:7] = qvel[:7]
    env.sim.forward()
    qpos_from_site_pose_kitchen(
        env,
        "end_effector",
        target_pos=target_pos,
        target_quat=target_quat.astype(np.float64),
        joint_names=[
            "panda0_joint0",
            "panda0_joint1",
            "panda0_joint2",
            "panda0_joint3",
            "panda0_joint4",
            "panda0_joint5",
            "panda0_joint6",
        ],
        tol=1e-14,
        rot_weight=1.0,
        regularization_threshold=0.1,
        regularization_strength=3e-2,
        max_update_norm=2.0,
        progress_thresh=20.0,
        max_steps=1000,
    )
    if is_grasped:
        env.sim.data.qpos[7:9] = gripper_qpos
        env.sim.data.qvel[7:9] = gripper_qvel

        # compute the transform between the old and new eef poses
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        ee_new_mat = pose2mat((env._eef_xpos, env._eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

        # apply the transform to the object
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pos, object_quat)))
        )
    #     set_object_pose(env, new_object_pose[0], new_object_pose[1], obj_idx=obj_idx)
    #     env.sim.forward()
    # else:
    #     # make sure the object is back where it started
    #     set_object_pose(env, object_pos, object_quat, obj_idx=obj_idx)

    if open_gripper_on_tp:
        env.sim.data.qpos[7:9] = np.array([0.04, 0.04])
        env.sim.data.qvel[7:9] = np.zeros(2)
        env.sim.forward()
    else:
        env.sim.data.qpos[7:9] = gripper_qpos
        env.sim.data.qvel[7:9] = gripper_qvel
        env.sim.forward()

    ee_error = np.linalg.norm(env._eef_xpos - target_pos)
    return ee_error


def check_robot_string(string):
    if string is None:
        return False
    return string.startswith("robot") or string.startswith("gripper")


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


def check_robot_collision(env, ignore_object_collision, obj_idx=0):
    # TODO: implement
    return False


def backtracking_search_from_goal(
    env,
    ik_ctrl,
    ignore_object_collision,
    start_pos,
    start_ori,
    goal_pos,
    ori,
    qpos,
    qvel,
    movement_fraction,
    is_grasped,
):
    # only search over the xyz position, orientation should be the same as commanded
    curr_pos = goal_pos.copy()
    set_robot_based_on_ee_pos(env, curr_pos, ori, ik_ctrl, qpos, qvel, is_grasped)
    collision = check_robot_collision(env, ignore_object_collision)
    iters = 0
    max_iters = int(1 / movement_fraction)
    while collision and iters < max_iters:
        curr_pos = curr_pos - movement_fraction * (goal_pos - start_pos)
        set_robot_based_on_ee_pos(
            env,
            curr_pos,
            ori,
            ik_ctrl,
            qpos,
            qvel,
            is_grasped,
        )
        collision = check_robot_collision(env, ignore_object_collision)
        iters += 1
    if collision:
        return np.concatenate(
            (start_pos, start_ori)
        )  # assumption is this is always valid!
    else:
        return np.concatenate((curr_pos, ori))


def update_controller_config(env, controller_config):
    controller_config["robot_name"] = env.robots[0].name
    controller_config["sim"] = env.robots[0].sim
    controller_config["eef_name"] = env.robots[0].gripper.important_sites["grip_site"]
    controller_config["eef_rot_offset"] = env.robots[0].eef_rot_offset
    controller_config["joint_indexes"] = {
        "joints": env.robots[0].joint_indexes,
        "qpos": env.robots[0]._ref_joint_pos_indexes,
        "qvel": env.robots[0]._ref_joint_vel_indexes,
    }
    controller_config["actuator_range"] = env.robots[0].torque_limits
    controller_config["policy_freq"] = env.robots[0].control_freq
    controller_config["ndim"] = len(env.robots[0].robot_joints)


def apply_controller(controller, action, robot, policy_step):
    gripper_action = None
    if robot.has_gripper:
        gripper_action = action[
            controller.control_dim :
        ]  # all indexes past controller dimension indexes
        arm_action = action[: controller.control_dim]
    else:
        arm_action = action

    # Update the controller goal if this is a new policy step
    if policy_step:
        controller.set_goal(arm_action)

    # Now run the controller for a step
    torques = controller.run_controller()

    # Clip the torques
    low, high = robot.torque_limits
    torques = np.clip(torques, low, high)

    # Get gripper action, if applicable
    if robot.has_gripper:
        robot.grip_action(gripper=robot.gripper, gripper_action=gripper_action)

    # Apply joint torque control
    robot.sim.data.ctrl[robot._ref_joint_actuator_indexes] = torques


def mp_to_point(
    env,
    ik_controller_config,
    osc_controller_config,
    pos,
    qpos,
    qvel,
    grasp=False,
    ignore_object_collision=False,
    planning_time=1,
    get_intermediate_frames=False,
    backtrack_movement_fraction=0.001,
):
    qpos_curr = env.sim.data.qpos.copy()
    qvel_curr = env.sim.data.qvel.copy()
    update_controller_config(env, ik_controller_config)
    ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
    ik_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)

    og_eef_xpos = env._eef_xpos.copy().astype(np.float64)
    og_eef_xquat = env._eef_xquat.copy().astype(np.float64)
    og_eef_xquat = og_eef_xquat / np.linalg.norm(og_eef_xquat)
    pos[3:] = pos[3:] / np.linalg.norm(pos[3:])

    def isStateValid(state):
        pos = np.array([state.getX(), state.getY(), state.getZ()])
        quat = np.array(
            [
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
        )
        if all(pos == og_eef_xpos) and all(quat == og_eef_xquat):
            # start state is always valid.
            return True
        else:
            # TODO; if it was grasping before ik and not after automatically set to invalid
            set_robot_based_on_ee_pos(
                env,
                pos,
                quat,
                ik_ctrl,
                qpos,
                qvel,
                grasp,
            )
            valid = not check_robot_collision(
                env, ignore_object_collision=ignore_object_collision
            )
            return valid

    # create an SE3 state space
    space = ob.SE3StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)

    # compare bounds to start state
    bounds_low = env.mp_bounds_low
    bounds_high = env.mp_bounds_high

    bounds_low = np.minimum(env.mp_bounds_low, og_eef_xpos)
    bounds_high = np.maximum(env.mp_bounds_high, og_eef_xpos)
    pos[:3] = np.clip(pos[:3], bounds_low, bounds_high)

    bounds.setLow(0, bounds_low[0])
    bounds.setLow(1, bounds_low[1])
    bounds.setLow(2, bounds_low[2])
    bounds.setHigh(0, bounds_high[0])
    bounds.setHigh(1, bounds_high[1])
    bounds.setHigh(2, bounds_high[2])
    space.setBounds(bounds)

    # construct an instance of space information from this state space
    si = ob.SpaceInformation(space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    # create a random start state
    start = ob.State(space)
    start().setXYZ(*og_eef_xpos)
    start().rotation().x = og_eef_xquat[0]
    start().rotation().y = og_eef_xquat[1]
    start().rotation().z = og_eef_xquat[2]
    start().rotation().w = og_eef_xquat[3]

    goal = ob.State(space)
    goal().setXYZ(*pos[:3])
    goal().rotation().x = pos[3]
    goal().rotation().y = pos[4]
    goal().rotation().z = pos[5]
    goal().rotation().w = pos[6]
    goal_valid = isStateValid(goal())
    goal_error = set_robot_based_on_ee_pos(
        env,
        pos[:3],
        pos[3:],
        ik_ctrl,
        qpos,
        qvel,
        grasp,
    )
    print(f"Goal Validity: {goal_valid}")
    print(f"Goal Error {goal_error}")
    if not goal_valid:
        pos = backtracking_search_from_goal(
            env,
            ik_ctrl,
            ignore_object_collision,
            og_eef_xpos,
            og_eef_xquat,
            pos[:3],
            pos[3:],
            qpos,
            qvel,
            is_grasped=grasp,
            movement_fraction=backtrack_movement_fraction,
        )
        goal = ob.State(space)
        goal().setXYZ(*pos[:3])
        goal().rotation().x = pos[3]
        goal().rotation().y = pos[4]
        goal().rotation().z = pos[5]
        goal().rotation().w = pos[6]
        goal_error = set_robot_based_on_ee_pos(
            env,
            pos[:3],
            pos[3:],
            ik_ctrl,
            qpos,
            qvel,
            grasp,
        )
        goal_valid = isStateValid(goal())
        print(f"Updated Goal Validity: {goal_valid}")
        print(f"Goal Error {goal_error}")
        if not goal_valid:
            cv2.imwrite(
                f"{logger.get_snapshot_dir()}/failed_{env.num_steps}.png",
                env.get_image(),
            )
    if grasp and get_intermediate_frames:
        print(f"Goal state has reward {env.reward(None)}")
    # create a problem instance
    pdef = ob.ProblemDefinition(si)
    # set the start and goal states
    pdef.setStartAndGoalStates(start, goal)
    # create a planner for the defined space
    planner = og.RRTConnect(si)
    # set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)
    planner.setRange(0.05)
    # perform setup steps for the planner
    planner.setup()
    # attempt to solve the problem within planning_time seconds of planning time
    solved = planner.solve(planning_time)

    if get_intermediate_frames:
        set_robot_based_on_ee_pos(
            env,
            og_eef_xpos,
            og_eef_xquat,
            ik_ctrl,
            qpos,
            qvel,
            grasp,
        )
        set_robot_based_on_ee_pos(
            env,
            pos[:3],
            pos[3:],
            ik_ctrl,
            qpos,
            qvel,
            grasp,
        )
    intermediate_frames = []
    if solved:
        path = pdef.getSolutionPath()
        success = og.PathSimplifier(si).simplify(path, 1.0)
        converted_path = []
        for s, state in enumerate(path.getStates()):
            new_state = [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
            if env.update_with_true_state:
                # get actual state that we used for collision checking on
                set_robot_based_on_ee_pos(
                    env,
                    new_state[:3],
                    new_state[3:],
                    ik_ctrl,
                    qpos,
                    qvel,
                    grasp,
                )
                new_state = np.concatenate((env._eef_xpos, env._eef_xquat))
            else:
                new_state = np.array(new_state)
            converted_path.append(new_state)
        # reset env to original qpos/qvel
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        env.sim.forward()

        update_controller_config(env, osc_controller_config)
        osc_ctrl = controller_factory("OSC_POSE", osc_controller_config)
        osc_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)
        osc_ctrl.reset_goal()
        for state in converted_path:
            desired_rot = quat2mat(state[3:])
            for _ in range(50):
                current_rot = quat2mat(env._eef_xquat)
                rot_delta = orientation_error(desired_rot, current_rot)
                pos_delta = state[:3] - env._eef_xpos
                if grasp:
                    grip_ctrl = env.grip_ctrl_scale
                else:
                    grip_ctrl = -1
                action = np.concatenate((pos_delta, rot_delta, [grip_ctrl]))
                if np.linalg.norm(action[:-4]) < 1e-3:
                    break
                policy_step = True
                for i in range(int(env.control_timestep / env.model_timestep)):
                    env.sim.forward()
                    apply_controller(osc_ctrl, action, env.robots[0], policy_step)
                    env.sim.step()
                    env._update_observables()
                    policy_step = False
                if hasattr(env, "num_steps"):
                    env.num_steps += 1
                if get_intermediate_frames:
                    im = env.get_image()
                    add_text(im, "Planner", (1, 10), 0.5, (0, 255, 0))
                    intermediate_frames.append(im)
        env.mp_mse = (
            np.linalg.norm(state - np.concatenate((env._eef_xpos, env._eef_xquat))) ** 2
        )
        print(f"Controller reaching MSE: {env.mp_mse}")
        env.goal_error = goal_error
    else:
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        env.sim.forward()
        env.mp_mse = 0
        env.goal_error = 0
        env.num_failed_solves += 1
    env.intermediate_frames = intermediate_frames
    return env._get_observations()


class KitchenEnv:
    def __init__(
        self,
        env,
        predict_done_actions=False,
        terminate_on_success=False,
        terminate_on_drop=False,
    ):
        # super().__init__(env)
        self._wrapped_env = env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space
        self.num_steps = 0
        self.predict_done_actions = predict_done_actions
        self.terminate_on_success = terminate_on_success
        self.terminate_on_drop = terminate_on_drop
        if self.predict_done_actions:
            self.action_space = spaces.Box(
                np.concatenate((self._wrapped_env.action_space.low, [-1])),
                np.concatenate((self._wrapped_env.action_space.high, [1])),
            )
        self.sim = self._wrapped_env.sim
        self.data = self._wrapped_env.data
        self.TASK_ELEMENTS = self._wrapped_env.TASK_ELEMENTS
        self.OBS_ELEMENT_INDICES = self._wrapped_env.OBS_ELEMENT_INDICES

    def get_observation(self):
        return np.zeros(
            self.observation_space.low.shape
        )  # only going to train from images

    def render(self, imwidth, imheight, mode="rgb_array"):
        return self._wrapped_env.render(mode=mode)

    def get_image(self):
        return self.render(
            mode="rgb_array",
            imwidth=512,
            imheight=512,
        )

    def reset(self, **kwargs):
        self.num_steps = 0
        self.was_in_hand = False
        self.has_succeeded = False
        self.terminal = False
        o = self._wrapped_env.reset(**kwargs)
        return self.get_observation()

    def check_grasp(
        self,
    ):
        return check_object_grasp(self)

    def update_done_info_based_on_termination(self, i, d):
        if self.terminal:
            # if we've already terminated, don't let the agent get any more reward
            i["bad_mask"] = 1
        else:
            i["bad_mask"] = 0
        if i["grasped"] and not self.was_in_hand:
            self.was_in_hand = True
        if self.was_in_hand and not i["grasped"] and self.terminate_on_drop:
            # if we've dropped the object, terminate
            self.terminal = True
        if i["success"] and self.terminate_on_success:
            self.has_succeeded = True
            self.terminal = True
        d = d or self.terminal
        return d

    def _check_success(self):
        return len(self._wrapped_env.tasks_to_complete) == 0

    def step(self, action):
        if self.predict_done_actions:
            old_action = action
            action = action[:-1]
        o, r, d, i = self._wrapped_env.step(action)
        self.num_steps += 1
        i["success"] = float(self._check_success())
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        if self.predict_done_actions:
            d = old_action[-1] > 0
        d = self.update_done_info_based_on_termination(i, d)
        return o, r, d, i

    def get_site_xpos(self, name):
        id = self._wrapped_env.sim.model.site_name2id(name)
        return self._wrapped_env.sim.data.site_xpos[id]

    def get_site_xmat(self, name):
        id = self._wrapped_env.sim.model.site_name2id(name)
        return self._wrapped_env.sim.data.site_xmat[id]

    def get_ee_pose(self):
        return self.get_site_xpos("end_effector")

    @property
    def _eef_xpos(self):
        return self.get_ee_pose().copy()

    @property
    def _eef_xquat(self):
        return T.convert_quat(
            T.mat2quat(self.get_site_xmat("end_effector").reshape(3, 3)), to="wxyz"
        )
        # return self.get_ee_quat()

    def set_state(self, qpos, qvel):
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()


class MPEnv(KitchenEnv):
    def __init__(
        self,
        env,
        controller_configs=None,
        recompute_reward_post_teleport=False,
        planner_command_orientation=False,
        num_ll_actions_per_hl_action=25,
        planner_only_actions=False,
        add_grasped_to_obs=False,
        terminate_on_last_state=False,
        # mp
        planning_time=1,
        mp_bounds_low=None,
        mp_bounds_high=None,
        update_with_true_state=False,
        grip_ctrl_scale=1,
        backtrack_movement_fraction=0.001,
        # teleport
        vertical_displacement=0.03,
        teleport_instead_of_mp=True,
        plan_to_learned_goals=False,
        learn_residual=False,
        clamp_actions=False,
        randomize_init_target_pos=False,
        randomize_init_target_pos_range=(0.04, 0.06),
        hardcoded_high_level_plan=True,
        use_teleports_in_step=True,
        hardcoded_orientations=False,
        # upstream env
        predict_done_actions=False,
        terminate_on_success=False,
        terminate_on_drop=False,
        # grasp checks
        verify_stable_grasp=False,
        reset_at_grasped_state=False,
    ):
        super().__init__(
            env,
            predict_done_actions=predict_done_actions,
            terminate_on_success=terminate_on_success,
            terminate_on_drop=terminate_on_drop,
        )
        self.num_steps = 0
        self.vertical_displacement = vertical_displacement
        self.teleport_instead_of_mp = teleport_instead_of_mp
        self.planning_time = planning_time
        self.plan_to_learned_goals = plan_to_learned_goals
        self.learn_residual = learn_residual
        self.mp_bounds_low = mp_bounds_low
        self.mp_bounds_high = mp_bounds_high
        self.update_with_true_state = update_with_true_state
        self.grip_ctrl_scale = grip_ctrl_scale
        self.clamp_actions = clamp_actions
        self.backtrack_movement_fraction = backtrack_movement_fraction
        self.randomize_init_target_pos = randomize_init_target_pos
        self.recompute_reward_post_teleport = recompute_reward_post_teleport
        self.controller_configs = controller_configs
        self.verify_stable_grasp = verify_stable_grasp
        self.randomize_init_target_pos_range = randomize_init_target_pos_range
        self.planner_command_orientation = planner_command_orientation
        self.num_ll_actions_per_hl_action = num_ll_actions_per_hl_action
        self.planner_only_actions = planner_only_actions
        self.add_grasped_to_obs = add_grasped_to_obs
        self.use_teleports_in_step = use_teleports_in_step
        self.take_planner_step = True
        self.current_ll_policy_steps = 0
        self.reset_at_grasped_state = reset_at_grasped_state
        self.terminate_on_last_state = terminate_on_last_state
        self.hardcoded_orientations = hardcoded_orientations
        self.hardcoded_high_level_plan = hardcoded_high_level_plan

        if self.add_grasped_to_obs:
            # update observation space
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_space.shape[0] + 1,),
            )

    def compute_hardcoded_orientation(self, target_pos, quat):
        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        # compute perpendicular top grasps for the object, pick one that has less error
        orig_ee_quat = self.reset_ori.copy()
        ee_euler = mat2euler(quat2mat(orig_ee_quat))
        obj_euler = mat2euler(quat2mat(quat))
        ee_euler[2] = obj_euler[2] + np.pi / 2
        target_quat1 = mat2quat(euler2mat(ee_euler))
        # error1 = set_robot_based_on_ee_pos(
        #     self,
        #     target_pos.copy(),
        #     target_quat1.copy(),
        #     self.ik_ctrl,
        #     self.reset_qpos,
        #     self.reset_qvel,
        #     is_grasped=False,
        #
        #     obj_idx=self.obj_idx,
        # )

        # ee_euler[2] = obj_euler[2] - np.pi / 2
        # target_quat2 = mat2quat(euler2mat(ee_euler))
        # error2 = set_robot_based_on_ee_pos(
        #     self,
        #     target_pos.copy(),
        #     target_quat2.copy(),
        #     self.ik_ctrl,
        #     self.reset_qpos,
        #     self.reset_qvel,
        #     is_grasped=False,
        #
        #     obj_idx=self.obj_idx,
        # )
        # if error1 < error2:
        #     target_quat = target_quat1
        # else:
        #     target_quat = target_quat2

        target_quat = target_quat1
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()
        return target_quat

    def get_target_pose_list(self):
        pose_list = []
        # init target pos (object pos + vertical displacement)
        # final target positions, depending on the task
        for idx, element in enumerate(self.TASK_ELEMENTS):
            object_pos, object_quat = get_object_pose_mp(self, obj_idx=idx)
            if element == "slide cabinet":
                target_pos = object_pos + np.array([0, -0.05, 0])
                target_quat = self.reset_ori
            elif element == "top burner":
                target_pos = object_pos + np.array([0, -0.05, 0])
                target_quat = self.reset_ori
            elif element == "hinge cabinet":
                target_pos = object_pos + np.array([0, -0.05, 0])
                target_quat = self.reset_ori
            elif element == "light switch":
                target_pos = object_pos + np.array([0, -0.05, 0])
                target_quat = self.reset_ori
            elif element == "microwave":
                target_pos = object_pos + np.array([0, -0.05, 0])
                target_quat = self.reset_ori
            elif element == "kettle":
                target_pos = object_pos + np.array([0, -0.05, 0])
                target_quat = self.reset_ori
            pose_list.append((target_pos, target_quat))
        return pose_list

    def get_target_pos(self):
        target_pose_list = self.get_target_pose_list()
        if self.high_level_step > len(target_pose_list) - 1:
            return target_pose_list[-1]
        return self.get_target_pose_list()[self.high_level_step]

    def reset(self, get_intermediate_frames=False, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self.ep_step_ctr = 0
        self.high_level_step = 0
        self.num_failed_solves = 0
        self.num_steps = 0
        self.current_ll_policy_steps = 0

        self.reset_pos = self._eef_xpos.copy()
        self.reset_ori = self._eef_xquat.copy()
        self.reset_qpos = self.sim.data.qpos.copy()
        self.reset_qvel = self.sim.data.qvel.copy()
        self.initial_object_pos = get_object_pose_mp(self)[0].copy()

        self.was_in_hand = False
        self.has_succeeded = False
        self.terminal = False
        self.take_planner_step = True

        self.teleport_on_grasp = True
        self.teleport_on_place = False
        if not self.plan_to_learned_goals and not self.planner_only_actions:
            target_pos, target_quat = self.get_target_pos()
            if self.teleport_instead_of_mp:
                set_robot_based_on_ee_pos(
                    self,
                    target_pos.copy(),
                    target_quat.copy(),
                    self.reset_qpos,
                    self.reset_qvel,
                    is_grasped=False,
                    obj_idx=self.obj_idx,
                    open_gripper_on_tp=True,
                )
                # self.num_steps += 100 #don't log this
            else:
                # TODO: have mp also open gripper here
                mp_to_point(
                    self,
                    self.ik_controller_config,
                    self.osc_controller_config,
                    np.concatenate((target_pos, target_quat)).astype(np.float64),
                    qpos=self.reset_qpos,
                    qvel=self.reset_qvel,
                    grasp=False,
                    planning_time=self.planning_time,
                    get_intermediate_frames=get_intermediate_frames,
                    backtrack_movement_fraction=self.backtrack_movement_fraction,
                )
            self.high_level_step += 1
            self.take_planner_step = False
        obs = self.get_observation()
        if self.add_grasped_to_obs:
            obs = np.concatenate((obs, np.array([0])))
        return obs

    @property
    def obj_idx(self):
        return self.high_level_step - 1

    def check_grasp(self, verify_stable_grasp=False):
        is_grasped = check_object_grasp(self, obj_idx=self.obj_idx)

        if is_grasped and verify_stable_grasp:
            obj_string = get_object_string(self, obj_idx=self.obj_idx)
            d = self.sim.data
            object_in_contact_with_env = False
            for coni in range(d.ncon):
                con1 = self.sim.model.geom_id2name(d.contact[coni].geom1)
                con2 = self.sim.model.geom_id2name(d.contact[coni].geom2)
                if not check_robot_string(con1) and check_string(con2, obj_string):
                    object_in_contact_with_env = True
                if not check_robot_string(con2) and check_string(con1, obj_string):
                    object_in_contact_with_env = True
            is_grasped = is_grasped and not object_in_contact_with_env
        return is_grasped

    def clamp_planner_action_mp_space_bounds(self, action):
        action[:3] = np.clip(action[:3], self.mp_bounds_low, self.mp_bounds_high)
        return action

    def step(self, action, get_intermediate_frames=False):
        if self.plan_to_learned_goals or self.planner_only_actions:
            if self.take_planner_step:
                target_pos = self.get_target_pos()
                if self.learn_residual:
                    pos = action[:3] + target_pos
                    if self.clamp_actions:
                        pos = self.clamp_planner_action_mp_space_bounds(pos)
                else:
                    pos = action[:3] + self._eef_xpos
                    if self.clamp_actions:
                        pos = self.clamp_planner_action_mp_space_bounds(pos)
                if self.planner_command_orientation:
                    rot_delta = euler2mat(action[3:6])
                    quat = mat2quat(rot_delta @ quat2mat(self.reset_ori))
                else:
                    quat = self.reset_ori
                action = action.astype(np.float64)
                # quat = quat / np.linalg.norm(quat) # might be necessary for MP code?
                is_grasped = self.check_grasp()
                if self.teleport_instead_of_mp:
                    # make gripper fully open at start
                    pos = backtracking_search_from_goal(
                        self,
                        self.ik_ctrl,
                        ignore_object_collision=is_grasped,
                        start_pos=self._eef_xpos,
                        start_ori=self._eef_xquat,
                        goal_pos=pos,
                        ori=quat,
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        movement_fraction=0.01,
                        is_grasped=is_grasped,
                    )
                    o = self._get_observations()
                else:
                    o = mp_to_point(
                        self,
                        self.ik_controller_config,
                        self.osc_controller_config,
                        np.concatenate((pos, quat), dtype=np.float64),
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        grasp=is_grasped,
                        ignore_object_collision=is_grasped,
                        planning_time=self.planning_time,
                        get_intermediate_frames=get_intermediate_frames,
                        backtrack_movement_fraction=self.backtrack_movement_fraction,
                    )
                o, r, d, i = self._flatten_obs(o), self.reward(action), False, {}
                self.take_planner_step = False
                self.high_level_step += 1
            else:
                o, r, d, i = self._wrapped_env.step(action)
                self.current_ll_policy_steps += 1
                if self.current_ll_policy_steps == self.num_ll_actions_per_hl_action:
                    self.take_planner_step = True
                    self.current_ll_policy_steps = 0
                self.num_steps += 1
            # print(self.take_planner_step, self.ep_step_ctr)
            self.ep_step_ctr += 1
        else:
            o, r, d, i = self._wrapped_env.step(action)
            self.current_ll_policy_steps += 1
            self.num_steps += 1
            self.ep_step_ctr += 1
            is_grasped = self.check_grasp(verify_stable_grasp=self.verify_stable_grasp)
            open_gripper_on_tp = False
            if self.hardcoded_high_level_plan:
                # check if sub-task is complete
                object_string = get_object_string(self, obj_idx=self.obj_idx)
                if object_string not in self._wrapped_env.tasks_to_complete:
                    open_gripper_on_tp = True
                    take_planner_step = True
                else:
                    take_planner_step = False
            else:
                take_planner_step = self.take_planner_step
            if self.high_level_step >= len(self.get_target_pose_list()):
                # at the final stage of the high level plan
                take_planner_step = False
            if take_planner_step:
                target_pos, target_quat = self.get_target_pos()
                if self.teleport_instead_of_mp:
                    set_robot_based_on_ee_pos(
                        self,
                        target_pos,
                        target_quat,
                        self.reset_qpos,
                        self.reset_qvel,
                        is_grasped=is_grasped,
                        obj_idx=self.obj_idx,
                        open_gripper_on_tp=open_gripper_on_tp,
                    )
                else:
                    # TODO: have mp also open gripper here if open_gripper_on_tp is True
                    mp_to_point(
                        self,
                        self.ik_controller_config,
                        self.osc_controller_config,
                        np.concatenate((target_pos, self.reset_ori)).astype(np.float64),
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        grasp=is_grasped,
                        ignore_object_collision=is_grasped,
                        planning_time=self.planning_time,
                        get_intermediate_frames=get_intermediate_frames,
                        backtrack_movement_fraction=self.backtrack_movement_fraction,
                    )
                # TODO: should re-compute reward here so it is clear what action caused high reward
                if self.recompute_reward_post_teleport:
                    r += self.env.reward()
                self.take_planner_step = False
                self.high_level_step += 1
            if self.current_ll_policy_steps == self.num_ll_actions_per_hl_action:
                self.take_planner_step = True
                self.current_ll_policy_steps = 0
        i["success"] = float(self._check_success())
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        if not self.teleport_instead_of_mp:
            # add in planner logs
            i["mp_mse"] = self.mp_mse
            i["num_failed_solves"] = self.num_failed_solves
            i["goal_error"] = self.goal_error
        if self.add_grasped_to_obs:
            o = np.concatenate((o, np.array([i["grasped"]])))
        if self.predict_done_actions:
            d = action[-1] > 0
        d = self.update_done_info_based_on_termination(i, d)
        if self.terminate_on_last_state:
            d = self.ep_step_ctr == self.horizon
        return o, r, d, i
