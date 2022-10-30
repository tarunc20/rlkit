import cv2
import numpy as np
from gym import spaces
from robosuite.controllers import controller_factory
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import (
    euler2mat,
    mat2quat,
    quat2mat,
    quat_conjugate,
    quat_multiply,
)

from rlkit.core import logger
from rlkit.envs.proxy_env import ProxyEnv
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

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou


def set_robot_based_on_ee_pos(
    env,
    target_pos,
    target_quat,
    ik,
    qpos,
    qvel,
    gripper_qpos_in=None,
    gripper_qvel_in=None,
    ee_to_object_translation=None,
    is_grasped=False,
):
    """
    Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
    If grasping an object, ensures the object moves with the arm in a consistent way.
    """
    # cache quantities from prior to setting the state
    object_quat = env.sim.data.qpos[19:23]
    gripper_qpos = env.sim.data.qpos[7:9]
    gripper_qvel = env.sim.data.qvel[7:9]

    # reset to canonical state before doing IK
    env.sim.data.qpos[:] = qpos
    env.sim.data.qvel[:] = qvel
    env.sim.forward()

    ik.sync_state()
    cur_rot_inv = quat_conjugate(env._eef_xquat.copy())
    rot_diff = quat2mat(quat_multiply(target_quat, cur_rot_inv))
    joint_pos = ik.joint_positions_for_eef_command(target_pos - env._eef_xpos, rot_diff)
    env.robots[0].set_robot_joint_positions(joint_pos)
    assert (
        env.sim.data.qpos[:7] - joint_pos
    ).sum() < 1e-10  # ensure we accurately set the sim pose to the ik command
    if is_grasped:
        env.sim.data.qpos[7:9] = gripper_qpos
        env.sim.data.qvel[7:9] = gripper_qvel
        if gripper_qpos_in is not None:
            env.sim.data.qpos[7:9] = gripper_qpos_in
        if gripper_qvel_in is not None:
            env.sim.data.qvel[7:9] = gripper_qvel_in
        if env.name.endswith("Lift"):
            env.sim.data.qpos[9:12] = env._eef_xpos + ee_to_object_translation
        elif env.name.endswith("PickPlaceBread"):
            env.sim.data.qpos[16:19] = env._eef_xpos + ee_to_object_translation
            env.sim.data.qpos[19:23] = quat_multiply(mat2quat(rot_diff), object_quat)
        env.sim.forward()


def check_robot_string(string):
    if string is None:
        return False
    return string.startswith("robot") or string.startswith("gripper")


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


def check_robot_collision(env, ignore_object_collision):
    if env.name.endswith("Lift"):
        obj_string = "cube"
    elif env.name.endswith("PickPlaceBread"):
        obj_string = "Bread"
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if check_robot_string(con1) ^ check_robot_string(con2):
            if (
                check_string(con1, obj_string)
                or check_string(con2, obj_string)
                and ignore_object_collision
            ):
                # if the robot and the object collide, then we can ignore the collision
                continue
            return True
        elif ignore_object_collision:
            if check_string(con1, obj_string) or check_string(con2, obj_string):
                # if we are supposed to be "ignoring object collisions" then we assume the
                # robot is "joined" to the object. so if the object collides with any non-robot
                # object, then we should call that a collision
                return True
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
    movement_fraction=0.001,
    ee_to_object_translation=None,
    is_grasped=False,
):
    # only search over the xyz position, orientation should be the same as commanded
    curr_pos = goal_pos.copy()
    set_robot_based_on_ee_pos(
        env, curr_pos, ori, ik_ctrl, qpos, qvel, ee_to_object_translation, is_grasped
    )
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
            ee_to_object_translation,
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

    og_eef_xpos = env._eef_xpos.copy()
    og_eef_xquat = env._eef_xquat.copy()
    if env.name.endswith("Lift"):
        ee_to_object_translation = (
            env.sim.data.body_xpos[env.cube_body_id] - og_eef_xpos
        )
    else:
        ee_to_object_translation = (
            env.sim.data.body_xpos[env.obj_body_id[env.obj_to_use]] - og_eef_xpos
        )

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
                env, pos, quat, ik_ctrl, qpos, qvel, ee_to_object_translation, grasp
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
        env, pos[:3], pos[3:], ik_ctrl, qpos, qvel, ee_to_object_translation, grasp
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
            ee_to_object_translation=ee_to_object_translation,
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
            ee_to_object_translation,
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
            ee_to_object_translation,
            grasp,
        )
        set_robot_based_on_ee_pos(
            env,
            pos[:3],
            pos[3:],
            ik_ctrl,
            qpos,
            qvel,
            ee_to_object_translation,
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
                    ee_to_object_translation,
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


class MPEnv(ProxyEnv):
    def __init__(
        self,
        env,
        vertical_displacement=0.03,
        teleport_position=True,
        planning_time=1,
        plan_to_learned_goals=False,
        execute_hardcoded_policy_to_goal=False,
        learn_residual=False,
        mp_bounds_low=None,
        mp_bounds_high=None,
        update_with_true_state=False,
        grip_ctrl_scale=1,
        clamp_actions=False,
        backtrack_movement_fraction=0.001,
        randomize_init_target_pos=False,
        teleport_on_grasp=False,
        slack_reward=0,
        predict_done_actions=False,
        terminate_on_success=False,
        run_controller_to_finish_place=True,
    ):
        super().__init__(env)
        self.add_cameras()
        self.num_steps = 0
        self.vertical_displacement = vertical_displacement
        self.teleport_position = teleport_position
        self.planning_time = planning_time
        self.plan_to_learned_goals = plan_to_learned_goals
        self.execute_hardcoded_policy_to_goal = execute_hardcoded_policy_to_goal
        self.learn_residual = learn_residual
        self.mp_bounds_low = mp_bounds_low
        self.mp_bounds_high = mp_bounds_high
        self.update_with_true_state = update_with_true_state
        self.grip_ctrl_scale = grip_ctrl_scale
        self.clamp_actions = clamp_actions
        self.backtrack_movement_fraction = backtrack_movement_fraction
        self.randomize_init_target_pos = randomize_init_target_pos
        self.teleport_on_grasp = teleport_on_grasp
        self.slack_reward = slack_reward
        self.predict_done_actions = predict_done_actions
        self.terminate_on_success = terminate_on_success
        self.run_controller_to_finish_place = run_controller_to_finish_place
        if self.predict_done_actions:
            self.action_space = spaces.Box(
                np.concatenate((self._wrapped_env.action_space.low, [-1])),
                np.concatenate((self._wrapped_env.action_space.high, [1])),
            )

    def add_cameras(self):
        for (cam_name, cam_w, cam_h, cam_d) in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
        ):
            # Add cameras associated to our arrays
            cam_sensors, _ = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                modality="image",
            )
            self.cam_sensor = cam_sensors

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def compute_ee_to_object_translation(self):
        if self.name.endswith("Lift"):
            return self.sim.data.body_xpos[self.cube_body_id] - self._eef_xpos
        else:
            return self.sim.data.qpos[16:19] - self._eef_xpos

    def get_init_target_pos(self):
        ee_to_object_translation = self.compute_ee_to_object_translation()
        if self.name.endswith("Lift"):
            pos = self.sim.data.body_xpos[self.cube_body_id]
        elif self.name.endswith("PickPlaceBread"):
            pos = self.sim.data.body_xpos[self.obj_body_id[self.obj_to_use]]
            self.target_z_pos = pos[-1] + 0.1
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
                scale = np.random.uniform(0.03, 0.06)
                shifted_pos = pos + random_perturbation * scale
                # backtrack from the position just in case we sampled a point in collision
                set_robot_based_on_ee_pos(
                    self,
                    shifted_pos.copy(),
                    self._eef_xquat,
                    self.ik_ctrl,
                    qpos,
                    qvel,
                    ee_to_object_translation=ee_to_object_translation,
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
            shifted_pos = pos + np.array([0, 0, self.vertical_displacement])
            set_robot_based_on_ee_pos(
                self,
                shifted_pos.copy(),
                self._eef_xquat,
                self.ik_ctrl,
                qpos,
                qvel,
                ee_to_object_translation=ee_to_object_translation,
            )
            assert not self.check_grasp()  # we should not cheat!
        # teleporting the arm can break the controller
        self.robots[0].controller.reset_goal()
        return pos

    def get_observation(self):
        di = self._wrapped_env._get_observations(force_update=True)
        return self._wrapped_env._flatten_obs(di)

    def reset(self, get_intermediate_frames=False, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self.ik_controller_config = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "converge_steps": 100,
        }
        self.osc_controller_config = {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "output_min": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        self.ep_step_ctr = 0
        self.num_failed_solves = 0
        self.num_steps = 0
        self.reset_pos = self._eef_xpos.copy()
        self.reset_ori = self._eef_xquat.copy()
        self.reset_qpos = self.sim.data.qpos.copy()
        self.reset_qvel = self.sim.data.qvel.copy()
        if not self.plan_to_learned_goals:
            if self.teleport_position:
                update_controller_config(self, self.ik_controller_config)
                self.ik_ctrl = controller_factory("IK_POSE", self.ik_controller_config)
                self.ik_ctrl.update_base_pose(
                    self.robots[0].base_pos, self.robots[0].base_ori
                )
                pos = self.get_init_target_pos()
                obs = self.get_observation()
                # self.num_steps += 100 #don't log this
            else:
                pos = self.get_init_target_pos()
                pos = np.concatenate((pos, self.reset_ori))
                obs = mp_to_point(
                    self,
                    self.ik_controller_config,
                    self.osc_controller_config,
                    pos.astype(np.float64),
                    qpos=self.reset_qpos,
                    qvel=self.reset_qvel,
                    grasp=False,
                    planning_time=self.planning_time,
                    get_intermediate_frames=get_intermediate_frames,
                    backtrack_movement_fraction=self.backtrack_movement_fraction,
                )
                obs = self._flatten_obs(obs)
        return obs

    def check_grasp(self):
        if self.name.endswith("Lift"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )
        elif self.name.endswith("PickPlaceBread"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.objects[self.object_id],
            )
        return is_grasped

    def get_target_pos(
        self,
    ):
        if self.name.endswith("Lift"):
            pose = np.array([0, 0, 0.05]) + self._eef_xpos
        elif self.name.endswith("PickPlaceBread"):
            pose = np.array(
                [
                    0.2,
                    0.15,
                    self.target_z_pos,
                ]
            )
        return pose

    def clamp_planner_action_mp_space_bounds(self, action):
        action[:3] = np.clip(action[:3], self.mp_bounds_low, self.mp_bounds_high)
        return action

    def step(self, action, get_intermediate_frames=False):
        if self.plan_to_learned_goals:
            if self.ep_step_ctr == 0 or self.ep_step_ctr == self.horizon + 1:
                if self.learn_residual:
                    pos = action[:3] + self.get_init_target_pos()
                    rot_delta = euler2mat(pos[3:6])
                    quat = mat2quat(rot_delta @ quat2mat(self.reset_ori))
                else:
                    if self.clamp_actions:
                        action = self.clamp_planner_action_mp_space_bounds(action)
                    action = action.astype(np.float64)
                    pos = action[:3]
                    # quat = mat2quat(euler2mat(action[3:6])).astype(np.float64)
                    # quat = quat / np.linalg.norm(quat)
                    quat = self._eef_xquat
                is_grasped = self.check_grasp()
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
                o, r, d, i = self._flatten_obs(o), self.reward(action), {}, False
            else:
                o, r, d, i = self._wrapped_env.step(action)
                self.num_steps += 1
            self.ep_step_ctr += 1
        else:
            o, r, d, i = self._wrapped_env.step(action)
            self.num_steps += 1
            self.ep_step_ctr += 1
            if self.ep_step_ctr == self.horizon or (
                self.teleport_on_grasp and self.check_grasp()
            ):
                is_grasped = self.check_grasp()
                target_pos = self.get_target_pos()
                ee_to_object_translation = (
                    self.compute_ee_to_object_translation().copy()
                )
                if self.teleport_position:
                    set_robot_based_on_ee_pos(
                        self,
                        target_pos,
                        self.reset_ori,
                        self.ik_ctrl,
                        self.reset_qpos,
                        self.reset_qvel,
                        gripper_qpos_in=self.sim.data.qpos[7:9].copy(),
                        gripper_qvel_in=self.sim.data.qvel[7:9].copy(),
                        ee_to_object_translation=ee_to_object_translation,
                        is_grasped=is_grasped,
                    )
                    # teleporting the arm can break the controller
                    self.robots[0].controller.reset_goal()
                    if (
                        self.name.endswith("PickPlaceBread")
                        and self.run_controller_to_finish_place
                    ):
                        for _ in range(30):
                            self.robots[
                                0
                            ].controller.reset_goal()  # teleporting the arm can break the controller
                            action = np.zeros(7)
                            action[-1] = -1
                            self.env.step(action)
                        if is_grasped and not self._check_success():
                            print("FAILED TO DROP THE CUBE")
                            print()
                            print()
                else:
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
        i["success"] = float(self._check_success())
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        if not self.teleport_position:
            i["mp_mse"] = self.mp_mse
            i["num_failed_solves"] = self.num_failed_solves
            i["goal_error"] = self.goal_error
        r += self.slack_reward
        if self.predict_done_actions:
            d = action[-1] > 0
        if self.terminate_on_success:
            d = float(self._check_success())
        return o, r, d, i


class RobosuiteEnv(ProxyEnv):
    def __init__(
        self,
        env,
        slack_reward=0,
        predict_done_actions=False,
        terminate_on_success=False,
    ):
        super().__init__(env)
        for (cam_name, cam_w, cam_h, cam_d) in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
        ):

            # Add cameras associated to our arrays
            cam_sensors, cam_sensor_names = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                modality="image",
            )
            self.cam_sensor = cam_sensors
        self.num_steps = 0
        self.slack_reward = slack_reward
        self.predict_done_actions = predict_done_actions
        self.terminate_on_success = terminate_on_success
        if self.predict_done_actions:
            self.action_space = spaces.Box(
                np.concatenate((self._wrapped_env.action_space.low, [-1])),
                np.concatenate((self._wrapped_env.action_space.high, [1])),
            )

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def reset(self, **kwargs):
        self.num_steps = 0
        return super().reset(**kwargs)

    def check_grasp(
        self,
    ):
        if self.name.endswith("Lift"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )
        elif self.name.endswith("PickPlaceBread"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.objects[self.object_id],
            )
        return is_grasped

    def step(self, action):
        if self.predict_done_actions:
            old_action = action
            action = action[:-1]
        o, r, d, i = super().step(action)
        self.num_steps += 1
        i["success"] = float(self._check_success())
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        r += self.slack_reward
        if self.predict_done_actions:
            d = old_action[-1] > 0
        if self.terminate_on_success:
            d = float(self._check_success())
        if self.num_steps == self.horizon:
            d = True
        return o, r, d, i
