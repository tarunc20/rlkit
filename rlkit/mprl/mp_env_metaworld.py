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
from robosuite.utils.transform_utils import (
    euler2mat,
    mat2pose,
    mat2quat,
    pose2mat,
    quat2mat,
    quat_conjugate,
    quat_multiply,
    convert_quat
)
import robosuite.utils.camera_utils as CU

from rlkit.core import logger
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.mprl import module
from rlkit.mprl.inverse_kinematics import qpos_from_site_pose
from rlkit.torch.model_based.dreamer.visualization import add_text
import matplotlib.pyplot as plt

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

def get_obj_name(env):
    if env.name == "hammer-v2":
        obj_name = "hammer"
    elif env.name == "assembly-v2" or env.name == "disassemble-v2":
        obj_name = "asmbly_peg"
    elif env.name == "peg-insert-side-v2":
        obj_name = "peg"
    else:
        obj_name = "obj"
    return obj_name

ROBOT_BODIES = [
    'right_hand', 'hand', 'rightclaw', 'rightpad', 'leftclaw', 'leftpad',
]

ENV_CAMERAS = {
    "assembly-v2":"corner",
    "hammer-v2":"corner3",
    "disassemble-v2":"corner",
    "stick-pull-v2":"corner2",
    "peg-insert-side-v2":"corner2"
}

def save_img(env, filename):
    import matplotlib.pyplot as plt 
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig(filename)
    plt.close()

def get_object_pos(env):
    """
    Note this is only used for computing the target for MP
    this is NOT the true object pose
    """
    if env.name == "hammer-v2":
        object_pos = env._get_pos_objects()[:3] + np.array([0.09, -0.02, 0.02])
    elif env.name == "stick-pull-v2":
        object_pos = env._get_pos_objects()[:3] + np.array([-0.02, -0.0, -0.01])
    elif env.name == "assembly-v2":
        object_pos = env._get_pos_objects().copy() + np.array([0.00, 0.01, 0.]) # 0.01 is to fix shifted pos
    elif env.name == "disassemble-v2":
        object_pos = env._get_pos_objects().copy() + np.array([0., 0., -0.03])
    elif env.name == "peg-insert-side-v2":
        object_pos = env._get_pos_objects().copy() + np.array([0.08, -0.0, 0.0])
    else:
        object_pos = env._get_pos_objects().copy()
    return object_pos


def get_object_pose(env):
    if env.name == "hammer-v2" or env.name == "stick-pull-v2":
        object_pos = env._get_pos_objects()[:3]
        object_quat = env._get_quat_objects()[:4]
    elif env.name == "assembly-v2" or env.name == "disassemble-v2":
        object_pos = env._get_pos_objects().copy() - np.array([0.13, 0., 0.,])
        object_quat = env._get_quat_objects().copy()
    else:
        object_pos = env._get_pos_objects().copy()
        object_quat = env._get_quat_objects().copy()
        # object_pos = env.sim.data.qpos[9:12].copy()
        # object_quat = env.sim.data.qpos[12:16].copy()
    return np.concatenate((object_pos, object_quat))


def set_object_pose(env, object_pos, object_quat):
    """
    Set the object pose in the environment.
    Makes sure to convert from xyzw to wxyz format for quaternion. qpos requires wxyz!
    Arguments:
        env
        object_pos (np.ndarray): 3D position of the object
        object_quat (np.ndarray): 4D quaternion of the object (xyzw format)

    """
    #object_quat = T.convert_quat(object_quat, to="wxyz")
    env._set_obj_pose(np.concatenate((object_pos, object_quat)))


def get_object_string(env):
    obj_string = "obj"
    return obj_string


def check_robot_string(string):
    if string is None:
        return False
    return (
        string.startswith("robot")
        or string.startswith("leftclaw")
        or string.startswith("rightclaw")
        or string.startswith("rightpad")
        or string.startswith("leftpad")
    )


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


def check_robot_collision(env, ignore_object_collision, verbose=False):
    obj_string = get_object_string(env)
    obj_name = get_obj_name(env)
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        body1 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom1]
        )
        body2 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom2]
        )
        if verbose:
            print(f"Con1: {con1} Con2: {con2} Body1: {body1} Body2: {body2}")
        if (check_robot_string(con1) ^ check_robot_string(con2)) or ((body1 in ROBOT_BODIES) ^ (body2 in ROBOT_BODIES)):
            if (
                check_string(con1, obj_string)
                or check_string(con2, obj_string)
                and ignore_object_collision
            ):
                # if the robot and the object collide, then we can ignore the collision
                continue
            # check using bodies
            if ((body1 == obj_name and body2 in ROBOT_BODIES) or (body2 == obj_name and body1 in ROBOT_BODIES)) and ignore_object_collision: # used to be not ignore robot collision
                continue 
            return True
        elif ignore_object_collision:
            if check_string(con1, obj_string) or check_string(con2, obj_string):
                # if we are supposed to be "ignoring object collisions" then we assume the
                # robot is "joined" to the object. so if the object collides with any non-robot
                # object, then we should call that a collision
                return True
            if (body1 == obj_name and body2 not in ROBOT_BODIES) or (body2 == obj_name and body1 not in ROBOT_BODIES):
                return True 
    return False


def gripper_contact(string, side):
    if string is None:
        return False
    if side == "left":
        return string.startswith("leftclaw") or string.startswith("leftpad")
    elif side == "right":
        return string.startswith("rightclaw") or string.startswith("rightpad")


def check_object_grasp(env):
    # TODO: finish this function
    obs = env._get_obs()
    obj = obs[4:7]
    action = np.zeros(4)
    action[-1] = 1
    # object_grasped = env._gripper_caging_reward(
    #     action,
    #     obj,
    #     obj_radius=0.015,
    #     pad_success_thresh=0.05,
    #     object_reach_radius=0.01,
    #     xz_thresh=0.01,
    #     desired_gripper_effort=0.7,
    #     high_density=True,
    # )
    thresh = 0.9
    # also check that object is in contact with gripper
    object_gripper_contact = False
    d = env.sim.data
    obj_string = get_object_string(env)
    left_gripper_contact = False
    right_gripper_contact = False
    object_in_contact_with_env = False
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if (gripper_contact(con1, "left") and check_string(con2, obj_string)) or (
            gripper_contact(con2, "left") and check_string(con1, obj_string)
        ):
            left_gripper_contact = True
        elif (gripper_contact(con1, "right") and check_string(con2, obj_string)) or (
            gripper_contact(con2, "right") and check_string(con1, obj_string)
        ):
            right_gripper_contact = True
        else:
            if not check_robot_string(con1) and check_string(con2, obj_string):
                object_in_contact_with_env = True
            if not check_robot_string(con2) and check_string(con1, obj_string):
                object_in_contact_with_env = True
        # if check_robot_string(con1) and check_string(con2, obj_string):
        #     object_gripper_contact = True
        # if check_robot_string(con2) and check_string(con1, obj_string):
        #     object_gripper_contact = True
    # check if there exists a string starting with left and a string starting with right in gripper contacts
    object_gripper_contact = left_gripper_contact and right_gripper_contact
    is_grasped = object_gripper_contact and (not object_in_contact_with_env)
    # is_grasped = object_grasped > thresh and object_gripper_contact and object_lifted
    return is_grasped

def body_check_grasp(env):
    # get correct object name 
    if env.name == "hammer-v2":
        obj_name = "hammer"
    elif env.name == "assembly-v2" or env.name == "disassemble-v2":
        obj_name = "asmbly_peg"
    elif env.name == "peg-insert-side-v2":
        obj_name = "peg"
    elif env.name == "stick-pull-v2":
        obj_name = None # hacky sol since the geoms and bodies are messed up for this environment
    else:
        return False
    object_gripper_contact = False
    d = env.sim.data
    obj_string = get_object_string(env)
    left_gripper_contact = False
    right_gripper_contact = False
    object_in_contact_with_env = False
    pad_names = ["leftpad", "rightpad"]
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        body1 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom1]
        )
        body2 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom2]
        )
        if body1 == "leftpad" and body2 == obj_name or body2 == "leftpad"and body1 == obj_name:
            left_gripper_contact = True 
        if body1 == "rightpad" and body2 == obj_name or body2 == "rightpad"and body1 == obj_name:
            right_gripper_contact = True 
    return left_gripper_contact and right_gripper_contact 

def set_robot_based_on_ee_pos(
    env,
    target_pos,
    target_quat,
    qpos,
    qvel,
    is_grasped,
):
    """
    Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
    If grasping an object, ensures the object moves with the arm in a consistent way.
    """
    # cache quantities from prior to setting the state
    object_pose = get_object_pose(env).copy()
    gripper_qpos = env.sim.data.qpos[7:9].copy()
    gripper_qvel = env.sim.data.qvel[7:9].copy()
    old_eef_xquat = env._eef_xquat.copy()
    old_eef_xpos = env._eef_xpos.copy()

    # reset to canonical state before doing IK
    env.sim.data.qpos[:7] = qpos[:7]
    env.sim.data.qvel[:7] = qvel[:7]
    env.sim.forward()
    result = qpos_from_site_pose(
        env,
        "endEffector",
        target_pos=target_pos,
        target_quat=target_quat.astype(np.float64),
        joint_names=[
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
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
            np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
        )
        set_object_pose(env, new_object_pose[0], new_object_pose[1])
        env.sim.forward()
    else:
        # make sure the object is back where it started
        set_object_pose(env, object_pose[:3], object_pose[3:])

    env.sim.data.qpos[7:9] = gripper_qpos
    env.sim.data.qvel[7:9] = gripper_qvel
    env.sim.forward()

    ee_error = np.linalg.norm(env._eef_xpos - target_pos)
    # need to update the mocap pos post teleport
    env.reset_mocap2body_xpos(env.sim)
    return ee_error


def backtracking_search_from_goal(
    env,
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
    set_robot_based_on_ee_pos(env, curr_pos, ori, qpos, qvel, is_grasped)
    collision = check_robot_collision(env, ignore_object_collision)
    iters = 0
    max_iters = int(1 / movement_fraction)
    while collision and iters < max_iters:
        curr_pos = curr_pos - movement_fraction * (goal_pos - start_pos)
        set_robot_based_on_ee_pos(
            env,
            curr_pos,
            ori,
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

# same as previous, but using geom id in case there is no geom name (needed for assembly task)
def get_geom_pose_from_seg(env, geom, camera_names, camera_width, camera_height, sim):
    pointclouds = []
    for camera_name in camera_names:
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        obj_mask = segmentation_map == geom
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
        pointclouds.append(obj_pointcloud)
    return np.mean(np.concatenate(pointclouds, axis=0), axis = 0)

def geom_pointcloud(env, geom, camera_names, camera_width, camera_height, sim):
    all_pts = []
    for camera_name in camera_names:
        sim = env.sim
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        obj_mask = segmentation_map == geom
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
        obj_pointcloud = CU.transform_from_pixels_to_world(
            pixels=np.argwhere(obj_mask),
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        all_pts.append(obj_pointcloud)
    return np.concatenate(all_pts)



################## VISION PIPELINE ##################


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
    # if (env._eef_xpos == pos[:3]).all():
    #     return 
    qpos_curr = env.sim.data.qpos.copy()
    qvel_curr = env.sim.data.qvel.copy()
    print(f"Starting eef xpos: {env._eef_xpos}")
    og_eef_xpos = env._eef_xpos.copy()
    og_eef_xquat = env._eef_xquat.copy()
    og_eef_xquat /= np.linalg.norm(og_eef_xquat)

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
            set_robot_based_on_ee_pos(env, pos, og_eef_xquat, qpos, qvel, grasp)
            valid = not check_robot_collision(
                env, ignore_object_collision=ignore_object_collision
            )
            return valid

    # create an SE3 state space
    space = ob.SE3StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)

    # compare bounds to start state
    bounds_low = np.array((-2., -2., -2.))#env.mp_bounds_low
    bounds_high = np.array((2., 2., 2.))#

    bounds_low = np.minimum(bounds_low, og_eef_xpos)
    bounds_high = np.maximum(bounds_high, og_eef_xpos)
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
    og_eef_xquat = og_eef_xquat.astype(np.float64)
    og_eef_xquat /= np.linalg.norm(og_eef_xquat)
    start = ob.State(space)
    start().setXYZ(*og_eef_xpos)
    start().rotation().x = og_eef_xquat[0]
    start().rotation().y = og_eef_xquat[1]
    start().rotation().z = og_eef_xquat[2]
    start().rotation().w = og_eef_xquat[3]

    goal = ob.State(space)
    goal().setXYZ(*pos[:3])
    goal().rotation().x = og_eef_xquat[0].astype(np.float64)
    goal().rotation().y = og_eef_xquat[1].astype(np.float64)
    goal().rotation().z = og_eef_xquat[2].astype(np.float64)
    goal().rotation().w = og_eef_xquat[3].astype(np.float64)
    goal_valid = isStateValid(goal())
    goal_error = set_robot_based_on_ee_pos(env, pos[:3], og_eef_xquat.copy(), qpos, qvel, grasp)
    check_robot_collision(env, ignore_object_collision=grasp, verbose=True)
    print(f"Goal Validity: {goal_valid}")
    print(f"Goal Error {goal_error}")
    print(f"Start valid: {isStateValid(start())}")
    # print(f"Start state: {start().getX(), start().getY(), start().getZ()}")
    # print(f"Space bounds: {space.getBounds().low[0], space.getBounds().low[1], space.getBounds().low[2]}")
    # print(f"Space bounds: {space.getBounds().high[0], space.getBounds().high[1], space.getBounds().high[2]}")
    # print(space.satisfiesBounds(start()))
    if not goal_valid:
        pos = backtracking_search_from_goal(
            env,
            ignore_object_collision,
            og_eef_xpos,
            og_eef_xquat,
            pos[:3],
            og_eef_xquat, # used to be pos[3:]
            qpos,
            qvel,
            is_grasped=grasp,
            movement_fraction=backtrack_movement_fraction,
        )
        goal = ob.State(space)
        goal().setXYZ(*pos[:3])
        goal().rotation().x = 0.
        goal().rotation().y = 0.
        goal().rotation().z = 0.
        goal().rotation().w = 1.
        goal_error = set_robot_based_on_ee_pos(
            env,
            pos[:3],
            pos[3:],
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
    # if get_intermediate_frames:
    #     set_robot_based_on_ee_pos(
    #         env,
    #         og_eef_xpos,
    #         og_eef_xquat,
    #         qpos,
    #         qvel,
    #         grasp,
    #     )
    #     set_robot_based_on_ee_pos(
    #         env,
    #         pos[:3],
    #         pos[3:],
    #         qpos,
    #         qvel,
    #         grasp,
    #     )
    intermediate_frames = []
    if solved:
        path = pdef.getSolutionPath()
        success = og.PathSimplifier(si).simplify(path, 1.0)
        converted_path = []
        for s, state in enumerate(path.getStates()):
            new_state = np.array([
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ])
            if env.update_with_true_state:
                # get actual state that we used for collision checking on
                set_robot_based_on_ee_pos(
                    env,
                    new_state[:3],
                    new_state[3:],
                    qpos,
                    qvel,
                    grasp,
                )
                new_state = np.concatenate((env._eef_xpos, env._eef_xquat))
            else:
                new_state = np.array(new_state)
            converted_path.append(new_state)
        converted_path = converted_path[1:]
        # reset env to original qpos/qvel
        waypoint_images = []
        waypoint_masks = []
        for i, state in enumerate(converted_path):
            set_robot_based_on_ee_pos(
                env,
                state[:3],
                state[3:],
                qpos,
                qvel,
                is_grasped=grasp,
            )
            im = env.sim.render(
                camera_name="corner",
                width=960,
                height=540,
            )
            # cv2.imwrite("test_{i}.png".format(i=i), im)
            sim = env.sim
            segmentation_map = np.flipud(CU.get_camera_segmentation(
                camera_name="corner",
                camera_width=960,
                camera_height=540,
                sim=sim,
            ))
            # get robot segmentation mask
            geom_ids = np.unique(segmentation_map[:, :, 1])
            robot_ids = []
            for geom_id in geom_ids:
                if geom_id != -1:
                    geom_name = sim.model.geom_id2name(geom_id)
                    if geom_name == None:
                        continue
                    if geom_name.startswith("robot") or geom_name.startswith("left") or geom_name.startswith("right") or geom_id == 27:
                        robot_ids.append(geom_id)
            robot_ids.append(27)
            robot_ids.append(28)
            robot_mask = np.expand_dims(np.any(
                [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids], axis=0
            ), -1)
            waypoint_masks.append(robot_mask)
            # cv2.imwrite('masked_test_{i}.png'.format(i=i), robot_mask*im)
            waypoint_images.append(robot_mask*im)
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        gripper_qpos = env.sim.data.qpos[7:9].copy()
        gripper_qvel = env.sim.data.qvel[7:9].copy()
        env.sim.step()
        env.sim.forward() 
        # return 
        env.set_robot_color(np.array([0.1, 0.3, 0.7, 1.0])) # replace later when generating videos
        for state_idx, state in enumerate(converted_path):
            desired_rot = quat2mat(state[3:])
            state_frames = []
            for _ in range(50):
                for s in range(50):
                    #env.sim.forward()
                    # set gripper qpos and qvel 
                    # replace with actually doing gripper action
                    env.set_xyz_action((state[:3] - env._eef_xpos))
                    if grasp:
                        env.do_simulation([1.0, -1.0], n_frames=env.frame_skip)
                    else:
                        env.do_simulation([0.0, -0.0], n_frames=env.frame_skip)
                    for site in env._target_site_config: # taken from metaworld repo
                        env._set_pos_site(*site)
                    # if get_intermediate_frames and s % 15 == 0:
                    #     im = env.sim.render(camera_name="corner", width=960, height=540).astype(np.float64)
                    #     im /= 255
                    #     robot_mask = waypoint_masks[state_idx].astype(np.float64)
                    #     robot_waypt = 0.5 * (waypoint_images[state_idx].astype(np.float64) / 255)
                    #     im = 0.5 * (im * robot_mask) + 0.5 * robot_waypt + im * (1 - robot_mask)
                    #     intermediate_frames.append((im * 255).astype(np.uint8))
                    if get_intermediate_frames and s % 15 == 0:
                        im = env.sim.render(camera_name="corner", width=960, height=540).astype(np.float64)
                        state_frames.append(im / 255)
                if hasattr(env, "num_steps"):
                    env.num_steps += 1
                if np.linalg.norm(state[:3] - env._eef_xpos) < 1e-5:
                    break
            # achieved state, now render and superimpose image over state 
            env.reset_robot_color()
            im = env.sim.render(
                camera_name="corner",
                width=960,
                height=540,
            )
            # cv2.imwrite("test_{i}.png".format(i=i), im)
            sim = env.sim
            segmentation_map = np.flipud(CU.get_camera_segmentation(
                camera_name="corner",
                camera_width=960,
                camera_height=540,
                sim=sim,
            ))
            # get robot segmentation mask
            geom_ids = np.unique(segmentation_map[:, :, 1])
            robot_ids = []
            for geom_id in geom_ids:
                if geom_id != -1:
                    geom_name = sim.model.geom_id2name(geom_id)
                    if geom_name == None:
                        continue
                    if geom_name.startswith("robot") or geom_name.startswith("left") or geom_name.startswith("right") or geom_id == 27:
                        robot_ids.append(geom_id)
            robot_ids.append(27)
            robot_ids.append(28)
            robot_mask = np.expand_dims(np.any(
                [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids], axis=0
            ), -1)
            waypoint_mask = robot_mask
            # cv2.imwrite('masked_test_{i}.png'.format(i=i), robot_mask*im)
            waypoint_img = robot_mask*im
            for i in range(len(state_frames)):
               robot_waypt = 0.5 * (waypoint_img.astype(np.float64) / 255) 
               robot_mask = waypoint_mask.astype(np.float64)
               state_frames[i] = 0.5 * (state_frames[i] * robot_mask) + 0.5 * robot_waypt + state_frames[i] * (1 - robot_mask)
               state_frames[i] = (state_frames[i] * 255).astype(np.uint8)
            #    print(state_frames[i].shape)
            env.set_robot_color(np.array([0.1, 0.3, 0.7, 1.0]))
            intermediate_frames.extend(state_frames)
        env.mp_mse = (
            np.linalg.norm(state - np.concatenate((env._eef_xpos, env._eef_xquat))) ** 2
        )
        print(f"Controller reaching MSE: {env.mp_mse}")
        env.goal_error = goal_error
        if get_intermediate_frames:
            print(f"Number of frames: {len(intermediate_frames)}")
            env.intermediate_frames = intermediate_frames
            env.reset_robot_color()
    else:
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        env.sim.forward()
        env.mp_mse = 0
        env.goal_error = 0
        env.num_failed_solves += 1
    env.intermediate_frames = intermediate_frames
    #return env._get_observations()


class MetaworldEnv(ProxyEnv):
    def __init__(
        self,
        env,
        slack_reward=0,
        predict_done_actions=False,
        terminate_on_success=False,
        terminate_on_drop=False,
    ):
        super().__init__(env)
        self.num_steps = 0
        self.slack_reward = slack_reward
        self.predict_done_actions = predict_done_actions
        self.terminate_on_success = terminate_on_success
        self.terminate_on_drop = terminate_on_drop
        if self.predict_done_actions:
            self.action_space = spaces.Box(
                np.concatenate((self._wrapped_env.action_space.low, [-1])),
                np.concatenate((self._wrapped_env.action_space.high, [1])),
            )

    def get_image(self):
        im = self.render(mode="rgb_array")[:, :, ::-1]
        return im

    def reset(self, **kwargs):
        self.num_steps = 0
        self.was_in_hand = False
        self.has_succeeded = False
        self.terminal = False
        return super().reset(**kwargs)

    def check_grasp(
        self,
    ):
        return check_object_grasp(self) or body_check_grasp(self)

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

    def step(self, action):
        if self.predict_done_actions:
            old_action = action
            action = action[:-1]
        o, r, d, i = super().step(action)
        self.num_steps += 1
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        r += self.slack_reward
        if self.predict_done_actions:
            d = old_action[-1] > 0
        return o, r, d, i

    @property
    def _eef_xpos(self):
        return self.get_endeff_pos()

    @property
    def _eef_xquat(self):
        return self.get_endeff_quat()


class MPEnv(MetaworldEnv):
    def __init__(
        self,
        env,
        name,
        controller_configs=None,
        recompute_reward_post_teleport=False,
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
        teleport_on_grasp=False,
        use_teleports_in_step=True,
        # vision stuff
        use_vision_pose_estimation=True,
        # upstream env
        slack_reward=0,
        predict_done_actions=False,
        terminate_on_success=False,
        terminate_on_drop=False,
        # grasp checks
        check_com_grasp=False,
        verify_stable_grasp=False,
        reset_at_grasped_state=False,
        max_path_length=200,
    ):
        super().__init__(
            env,
            slack_reward=slack_reward,
            predict_done_actions=predict_done_actions,
            terminate_on_success=terminate_on_success,
            terminate_on_drop=terminate_on_drop,
        )
        self.name = name
        self.num_steps = 0
        self.vertical_displacement = vertical_displacement
        self.teleport_instead_of_mp = teleport_instead_of_mp
        self.planning_time = planning_time
        # add more planning time
        self.planning_time = 10.0
        self.plan_to_learned_goals = plan_to_learned_goals
        self.learn_residual = learn_residual
        self.mp_bounds_low = mp_bounds_low
        self.mp_bounds_high = mp_bounds_high
        self.update_with_true_state = update_with_true_state
        self.grip_ctrl_scale = grip_ctrl_scale
        self.clamp_actions = clamp_actions
        self.backtrack_movement_fraction = backtrack_movement_fraction
        self.randomize_init_target_pos = randomize_init_target_pos
        self.teleport_on_grasp = teleport_on_grasp
        self.check_com_grasp = check_com_grasp
        self.recompute_reward_post_teleport = recompute_reward_post_teleport
        self.verify_stable_grasp = verify_stable_grasp
        self.randomize_init_target_pos_range = randomize_init_target_pos_range
        self.num_ll_actions_per_hl_action = num_ll_actions_per_hl_action
        self.planner_only_actions = planner_only_actions
        self.add_grasped_to_obs = add_grasped_to_obs
        self.use_teleports_in_step = use_teleports_in_step
        self.take_planner_step = True
        self.current_ll_policy_steps = 0
        self.reset_at_grasped_state = reset_at_grasped_state
        self.terminate_on_last_state = terminate_on_last_state
        self.planner_command_orientation = False
        self.max_path_length = max_path_length
        self.use_vision_pose_estimation = use_vision_pose_estimation

        if self.add_grasped_to_obs:
            # update observation space
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_space.shape[0] + 1,),
            )
        self.robot_bodies = [
            'base', 'controller_box', 
            'pedestal_feet', 'torso', 
            'pedestal', 'right_arm_base_link', 
            'right_l0', 'head', 
            'screen', 'head_camera', 
            'right_torso_itb', 'right_l1', 
            'right_l2', 'right_l3', 
            'right_l4', 'right_arm_itb', 
            'right_l5', 'right_hand_camera', 
            'right_wrist', 'right_l6', 'right_hand', 
            'hand', 'rightclaw', 'rightpad', 
            'leftclaw', 'leftpad', 'right_l4_2', 
            'right_l2_2', 'right_l1_2',
        ]
        self.robot_body_ids, self.robot_geom_ids = self.get_body_geom_ids_from_robot_bodies()
        self.original_colors = [env.sim.model.geom_rgba[idx].copy() for idx in self.robot_geom_ids]

    def get_body_geom_ids_from_robot_bodies(self):
        body_ids = [self.sim.model.body_name2id(body) for body in self.robot_bodies]
        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return body_ids, geom_ids

    def set_robot_color(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self.sim.model.geom_rgba[geom_id] = colors[idx]
        self.sim.forward()

    def reset_robot_color(self):
        self.set_robot_color(self.original_colors)
        self.sim.forward()

    def get_target_pos_list(self):
        pos_list = []
        # init target pos (object pos + vertical displacement)
        pos = get_object_pos(self)
        pos = pos + np.array([0, 0, self.vertical_displacement])
        pos_list.append(pos)
        # final target positions, depending on the task
        pos_list.append(self._target_pos + np.array([0, 0, 0.15]))
        return pos_list

    def get_target_pos(self):
        target_pos_list = self.get_target_pos_list()
        if self.high_level_step > len(target_pos_list) - 1:
            return target_pos_list[-1]
        return self.get_target_pos_list()[self.high_level_step]

    def get_init_target_pos(self):
        pos = get_object_pos(self)
        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        if self.randomize_init_target_pos:
            # sample a random position in a sphere around the target (not in collision)
            # the orientation of the arm should not be changed
            stop_sampling_target_pos = False
            xquat = self._eef_xquat
            while not stop_sampling_target_pos:
                random_perturbation = np.random.normal(0, 1, 3)
                random_perturbation[2] = np.abs(random_perturbation[2])
                random_perturbation /= np.linalg.norm(random_perturbation)
                scale = np.random.uniform(*self.randomize_init_target_pos_range)
                shifted_pos = pos + random_perturbation * scale
                # backtrack from the position just in case we sampled a point in collision
                set_robot_based_on_ee_pos(
                    self,
                    shifted_pos.copy(),
                    self._eef_xquat,
                    qpos,
                    qvel,
                    is_grasped=False,
                )
                ori_cond = np.linalg.norm(self._eef_xquat - xquat) < 1e-6
                grasp_cond = not self.check_grasp()
                collision_cond = not check_robot_collision(
                    self, ignore_object_collision=False
                )
                if ori_cond and grasp_cond and collision_cond:
                    stop_sampling_target_pos = True
                else:
                    self.sim.data.qpos[:] = qpos
                    self.sim.data.qvel[:] = qvel
                    self.sim.forward()
        else:
            if not self.use_vision_pose_estimation:
                shifted_pos = pos + np.array([0, -0.01, self.vertical_displacement])
                set_robot_based_on_ee_pos(
                    self,
                    shifted_pos.copy(),
                    self._eef_xquat,
                    qpos,
                    qvel,
                    is_grasped=False,
                )
            else:
                if self.name == "assembly-v2" or self.name == "disassemble-v2":
                    pos = get_geom_pose_from_seg(
                        self, 
                        self.sim.model.geom_name2id("WrenchHandle"), 
                        ["corner", "corner2"], 
                        500, 
                        500, 
                        self.sim
                        ) + np.array([0.02, 0.02, 0.03]) # used to be 0.02, 0.02, 0.03
                    if self.teleport_instead_of_mp:
                        set_robot_based_on_ee_pos(
                            self,
                            pos, 
                            self._eef_xquat,
                            qpos, 
                            qvel,
                            is_grasped=False,
                        )
                    else:
                        # set_robot_based_on_ee_pos(
                        #     self,
                        #     pos, 
                        #     self._eef_xquat,
                        #     qpos, 
                        #     qvel,
                        #     is_grasped=False,
                        # )
                        obs = mp_to_point(
                            self,
                            pos=np.concatenate((pos.astype(np.float64), self._eef_xquat)),
                            osc_controller_config=None,
                            ik_controller_config=None,
                            qpos=self.reset_qpos,
                            qvel=self.reset_qvel,
                            grasp=False,
                            planning_time=self.planning_time,
                            get_intermediate_frames=True,
                            backtrack_movement_fraction=self.backtrack_movement_fraction,
                        ) 
                if self.name == "hammer-v2":
                    obj_pose = get_geom_pose_from_seg(
                        self, 
                        self.sim.model.geom_name2id("HammerHandle"),
                        ["topview", "corner2"],
                        500,
                        500,
                        self.sim        
                    ) + np.array([0., 0., 0.05])
                    if self.teleport_instead_of_mp:
                        set_robot_based_on_ee_pos(
                            self, 
                            obj_pose,
                            self._eef_xquat,
                            qpos,
                            qvel,
                            False,
                        )
                    else:
                        obs = mp_to_point(
                            self,
                            pos=np.concatenate((obj_pose.astype(np.float64), self._eef_xquat)),
                            osc_controller_config=None,
                            ik_controller_config=None,
                            qpos=self.reset_qpos,
                            qvel=self.reset_qvel,
                            grasp=False,
                            planning_time=self.planning_time,
                            get_intermediate_frames=True,
                            backtrack_movement_fraction=self.backtrack_movement_fraction,
                        )
                if self.name == "peg-insert-side-v2":
                    obj_pose = get_geom_pose_from_seg(
                        self, 
                        self.sim.model.geom_name2id("peg"),
                        ["topview", "corner2"],
                        500,
                        500,
                        self.sim        
                    )
                    set_robot_based_on_ee_pos(
                        self, 
                        obj_pose + np.array([0.05, 0., 0.02]),
                        self._eef_xquat,
                        self.sim.data.qpos.copy(),
                        self.sim.data.qvel.copy(),
                        False,
                    )
                if self.name == "stick-pull-v2":
                    stick_pos = get_geom_pose_from_seg(
                        self, 
                        36,
                        ["topview", "corner2"],
                        500,
                        500,
                        self.sim        
                    ) + np.array([-0.05, 0., 0.02])
                    set_robot_based_on_ee_pos(
                        self, 
                        stick_pos,
                        self._eef_xquat,
                        qpos,
                        qvel,
                        False,
                    )
                if self.name == "bin-picking-v2":
                    obj_pose = get_geom_pose_from_seg(
                        self, 
                        36,
                        ["topview", "corner2"],
                        500,
                        500,
                        self.sim        
                    ) + np.array([-0.00, 0.0, 0.02])
                    if self.teleport_instead_of_mp:
                        set_robot_based_on_ee_pos(
                            self, 
                            obj_pose,
                            self._eef_xquat,
                            qpos,
                            qvel,
                            False,
                        )
                    else:
                        obs = mp_to_point(
                            self,
                            pos=np.concatenate((obj_pose.astype(np.float64), self._eef_xquat)),
                            osc_controller_config=None,
                            ik_controller_config=None,
                            qpos=self.reset_qpos,
                            qvel=self.reset_qvel,
                            grasp=False,
                            planning_time=self.planning_time,
                            get_intermediate_frames=True,
                            backtrack_movement_fraction=self.backtrack_movement_fraction,
                        )
        return pos

    def reset(self, get_intermediate_frames=False, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self.ep_step_ctr = 0
        self.high_level_step = 0
        self.num_failed_solves = 0
        self.num_steps = 0
        self.reset_pos = self._eef_xpos.copy()
        self.reset_ori = self._eef_xquat.copy()
        self.reset_qpos = self.sim.data.qpos.copy()
        self.reset_qvel = self.sim.data.qvel.copy()
        self.initial_object_pos = get_object_pos(self).copy()
        self.was_in_hand = False
        self.has_succeeded = False
        self.terminal = False
        self.take_planner_step = True
        self.current_ll_policy_steps = 0
        if not self.plan_to_learned_goals and not self.planner_only_actions:
            if self.teleport_instead_of_mp:
                pos = self.get_init_target_pos()
                obs = self._get_obs()
                # self.num_steps += 100 #don't log this
            else:
                pos = self.get_init_target_pos()
                pos = np.concatenate((pos, self.reset_ori))
                obs = self._get_obs()
        if self.reset_at_grasped_state:
            pos = self.get_init_target_pos()
            for i in range(15):
                a = np.concatenate(([0, 0, -0.3], [-1]))
                o, r, d, info = self._wrapped_env.step(a)
            for i in range(10):
                a = np.concatenate(([0, 0, 0], [1]))
                o, r, d, info = self._wrapped_env.step(a)
            if not self.check_grasp():
                print("Grasp failed, resetting")
                self.reset()
        self.hasnt_teleported = True
        if self.add_grasped_to_obs:
            obs = np.concatenate((obs, np.array([0])))
        return obs

    def check_grasp(self, verify_stable_grasp=False):
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        is_grasped = super().check_grasp()

        if is_grasped and verify_stable_grasp:
            # verify grasp is stable by lifting the arm and seeing if still in contact with object
            for i in range(10):
                action = np.zeros(4)
                action[2] = 0.1
                action[-1] = 1.0
                self._wrapped_env.step(action)
            is_grasped = super().check_grasp()
            self.sim.data.qpos[:] = qpos
            self.sim.data.qvel[:] = qvel
            self.sim.forward()

        if is_grasped and self.check_com_grasp:
            # check if left gripper pad is left of the com of object, right gripper pad is right of the com of object

            def name2id(type_name, name):
                obj_id = self.mjlib.mj_name2id(
                    self.model.ptr, self.mjlib.mju_str2Type(type_name), name.encode()
                )
                if obj_id < 0:
                    raise ValueError(
                        'No {} with name "{}" exists.'.format(type_name, name)
                    )
                return obj_id

            left_pos = self.sim.data.geom_xpos[
                name2id(
                    "geom", self.robots[0].gripper.important_geoms["left_fingerpad"][0]
                )
            ]
            right_pos = self.sim.data.geom_xpos[
                name2id(
                    "geom", self.robots[0].gripper.important_geoms["right_fingerpad"][0]
                )
            ]
            object_pos = get_object_pos(self)
            below_com_grasp = (left_pos[-1] - object_pos[-1] - 0.025) < 0 and (
                right_pos[-1] - object_pos[-1] - 0.025
            ) < 0
            if below_com_grasp:
                return True
            else:
                return False
        # print(f"Height increase: {get_object_pos(self)[2] - self.initial_object_pos[2]}")
        # print(f"Grasped: {(is_grasped and not self.check_com_grasp) and get_object_pos(self)[2] - self.initial_object_pos[2] > 0.04 }")
        return (is_grasped and not self.check_com_grasp) and \
                get_object_pos(self)[2] - self.initial_object_pos[2] > 0.04 # should be 0.02 when testing with policies but try both

    def get_target_pos_no_planner(
        self,
    ):
        try:
            if self.name == "peg-insert-side-v2":
                if self.use_vision_pose_estimation:
                    pose = self._wrapped_env.sim.data.get_site_xpos("hole") + np.array([0.25, 0., 0.045])
                else:
                    pose = self._wrapped_env.sim.data.get_site_xpos("hole") + np.array([0.25, 0., 0.045])
            elif self.name == "hammer-v2":
                if self.use_vision_pose_estimation:
                    pose = get_geom_pose_from_seg(
                        self,
                        53,
                        ["corner", "corner2","topview", "corner3"],
                        500,
                        500,
                        self.sim
                    ) 
                    pose += np.array([-0.14, -0.17, 0.05])
                else:
                    pose = self._wrapped_env._get_pos_objects()[3:] + np.array([-0.05, -0.20, 0.05])
            elif self.name == "assembly-v2":
                if self.use_vision_pose_estimation:
                    pose = get_geom_pose_from_seg(
                        self,
                        49,
                        ["corner", "corner2"],
                        500,
                        500,
                        self.sim
                    ) 
                    pose += np.array([0.13, 0.0, 0.15]) # go back to 0.12
                else:
                    raise NotImplementedError
            elif self.name == "stick-pull-v2":
                pail_pos = stick_pos = get_geom_pose_from_seg(
                    self, 
                    39,
                    ["corner", "corner2"],
                    500,
                    500,
                    self.sim        
                )
                pose = pail_pos + np.array([-0.13, -0.05, -0.02])
            elif self.name == "bin-picking-v2":
                pose = get_geom_pose_from_seg(
                        self,
                        44,
                        ["corner", "corner2"],
                        500,
                        500,
                        self.sim
                    ) + np.array([0.03, 0.03, 0.10])
            else: # bin picking 
                pose = None 
                #pose = self._target_pos + np.array([0, 0, 0.15])
        except:
            pose = self._eef_xpos
        return pose

    def clamp_planner_action_mp_space_bounds(self, action):
        action[:3] = np.clip(action[:3], self.mp_bounds_low, self.mp_bounds_high)
        return action

    def step(self, action, get_intermediate_frames=False):
        #self.intermediate_frames = []
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
                    self.hasnt_teleported = False
                    o = self._get_obs()
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
                        get_intermediate_frames=True,
                        backtrack_movement_fraction=self.backtrack_movement_fraction,
                    )
                    self.hasnt_teleported = False
                    o = self._get_obs()
                r, i = self.evaluate_state(o, action)
                r = r * self.reward_scale
                d = False
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
            curr_len = self._wrapped_env.curr_path_length
            o, r, d, i = self._wrapped_env.step(action)
            print(f"Current env eef xpos: {self._eef_xpos}")
            self.num_steps += 1
            self.ep_step_ctr += 1
            if self.hasnt_teleported:
                is_grasped = self.check_grasp(
                    verify_stable_grasp=self.verify_stable_grasp
                )
                #print(f"Is grasped :{is_grasped}")
            else:
                is_grasped = False
            if (self.teleport_on_grasp and is_grasped) and self.use_teleports_in_step and self.name != "disassemble-v2":
                target_pos = self.get_target_pos_no_planner()
                print(f"Target pos: {target_pos}")
                if self.teleport_instead_of_mp:
                    set_robot_based_on_ee_pos(
                        self,
                        target_pos,
                        self.reset_ori,
                        self.reset_qpos,
                        self.reset_qvel,
                        is_grasped=is_grasped,
                    )
                    self.hasnt_teleported = False
                    print(
                        "distance to goal: ",
                        np.linalg.norm(target_pos - self._eef_xpos),
                    )
                else:
                    obs = mp_to_point(
                        self,
                        None,
                        None,
                        np.concatenate((target_pos, self.reset_ori)).astype(np.float64),
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        grasp=is_grasped,
                        ignore_object_collision=is_grasped, # figure out this difference 
                        planning_time=self.planning_time,
                        get_intermediate_frames= True,#get_intermediate_frames,
                        backtrack_movement_fraction=self.backtrack_movement_fraction,
                    )
                    self.hasnt_teleported = False
                # TODO: should re-compute reward here so it is clear what action caused high reward
                if self.recompute_reward_post_teleport:
                    r += self.env.reward()
            #assert self._wrapped_env.curr_path_length == curr_len + 1
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        # if not self.teleport_instead_of_mp:
        #     # add in planner logs
        #     i["mp_mse"] = self.mp_mse
        #     i["num_failed_solves"] = self.num_failed_solves
        #     i["goal_error"] = self.goal_error
        if self.add_grasped_to_obs:
            o = np.concatenate((o, np.array([i["grasped"]])))
        r += self.slack_reward
        if self.predict_done_actions:
            d = action[-1] > 0
        d = self.update_done_info_based_on_termination(i, d)
        if self.terminate_on_last_state:
            d = self.ep_step_ctr == self.horizon
        #print(f"Num intermediate frames: {len(self.intermediate_frames)}")
        return o, r, d, i
