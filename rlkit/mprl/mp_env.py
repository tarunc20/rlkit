import cv2
import numpy as np
from robosuite.controllers import controller_factory
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import quat2mat

from rlkit.envs.proxy_env import ProxyEnv

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import geometric as og


def set_robot_based_on_ee_pos(env, pos, quat, ctrl):
    ctrl.sync_state()
    desired_rot = quat2mat(quat)
    cur_rot = quat2mat(env._eef_xquat)
    rot_diff = desired_rot @ np.linalg.inv(cur_rot)
    joint_pos = ctrl.joint_positions_for_eef_command(pos-env._eef_xpos, rot_diff)
    env.robots[0].set_robot_joint_positions(joint_pos)

def check_robot_string(string):
    return string.startswith("robot") or string.startswith("gripper")


def check_robot_collision(env, ignore_object_collision):
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if con1 is not None and con2 is not None:
            if check_robot_string(con1) or check_robot_string(con2):
                if env.name.endswith("Lift"):
                    obj_string = "cube"
                elif env.name.endswith("PickPlaceCan"):
                    obj_string = "Can"
                if (
                    con1.startswith(obj_string)
                    or con2.startswith(obj_string)
                    and ignore_object_collision
                ):
                    continue
                return True
    return False


def check_robot_collision_with_cube(env):
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if check_robot_string(con1) or check_robot_string(con2):
            if con1.startswith("cube") or con2.startswith("cube"):
                return True
    return False


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
    env, ik_ctrl, osc_ctrl, pos, grasp=False, ignore_object_collision=False
):
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
        set_robot_based_on_ee_pos(env, pos, quat, ik_ctrl)
        valid = not check_robot_collision(
            env, ignore_object_collision=ignore_object_collision
        )
        return valid

    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()
    og_eef_xpos = env._eef_xpos
    # create an SE3 state space
    space = ob.SE3StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-1.5)
    bounds.setHigh(1.5)
    space.setBounds(bounds)

    # construct an instance of space information from this state space
    si = ob.SpaceInformation(space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    # create a random start state
    start = ob.State(space)
    start().setXYZ(*env._eef_xpos)
    start().rotation().x = env._eef_xquat[0]
    start().rotation().y = env._eef_xquat[1]
    start().rotation().z = env._eef_xquat[2]
    start().rotation().w = env._eef_xquat[3]
    # create a random goal state
    goal = ob.State(space)
    goal().setXYZ(*pos[:3])
    goal().rotation().x = pos[3]
    goal().rotation().y = pos[4]
    goal().rotation().z = pos[5]
    goal().rotation().w = pos[6]
    print(
        f"State validity checks: Start: {isStateValid(start())}, Goal: {isStateValid(goal())}"
    )
    # create a problem instance
    pdef = ob.ProblemDefinition(si)
    # set the start and goal states
    pdef.setStartAndGoalStates(start, goal)
    # create a planner for the defined space
    planner = og.RRTstar(si)
    # set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)
    # perform setup steps for the planner
    planner.setup()
    # attempt to solve the problem within one second of planning time
    solved = planner.solve(1)

    if solved:
        # reset env to original qpos/qvel
        env.sim.data.qpos[:] = qpos.copy()
        env.sim.data.qvel[:] = qvel.copy()
        env.sim.forward()
        assert (env._eef_xpos == og_eef_xpos).all()

        path = pdef.getSolutionPath()

        converted_path = []
        for state in path.getStates():
            new_state = [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
            converted_path.append(new_state)
        osc_ctrl.reset_goal()
        for state in converted_path:
            state = np.array(state)
            desired_rot = quat2mat(state[3:])
            for _ in range(100):
                current_rot = quat2mat(env._eef_xquat)
                rot_delta = orientation_error(desired_rot, current_rot)
                pos_delta = state[:3] - env._eef_xpos
                if grasp:
                    grip_ctrl = 1
                else:
                    grip_ctrl = 0
                action = np.concatenate((pos_delta, rot_delta, [grip_ctrl]))
                if np.linalg.norm(action[:-1]) < 1e-5:
                    break
                # obs = env.wrapped_env.step(action)[0]
                policy_step = True
                for i in range(int(env.control_timestep / env.model_timestep)):
                    env.sim.forward()
                    apply_controller(osc_ctrl, action, env.robots[0], policy_step)
                    env.sim.step()
                    env._update_observables()
                    policy_step = False
                if hasattr(env, "num_steps"):
                    env.num_steps += 1
        env.mp_init_mse = (
            np.linalg.norm(state - np.concatenate((env._eef_xpos, env._eef_xquat))) ** 2
        )
    return env._get_observations()


class MPEnv(ProxyEnv):
    def __init__(self, env, vertical_displacement, teleport_position=True):
        super().__init__(env)
        for (cam_name, cam_w, cam_h, cam_d) in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
            # self.camera_segmentations,
        ):

            # Add cameras associated to our arrays
            cam_sensors, cam_sensor_names = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                # cam_segs=cam_segs,
                modality="image",
            )
            self.cam_sensor = cam_sensors
        self.num_steps = 0
        self.vertical_displacement = vertical_displacement
        self.teleport_position = teleport_position

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def get_init_target_pos(self):
        if self.name.endswith("Lift"):
            pos = self.sim.data.body_xpos[self.cube_body_id]
        elif self.name.endswith("PickPlaceCan"):
            pos = self.sim.data.body_xpos[self.obj_body_id[self.obj_to_use]]
        pos += np.array([0, 0, self.vertical_displacement])
        return pos

    def reset(self, **kwargs):
        self._wrapped_env.reset(**kwargs)
        ik_controller_config = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        osc_controller_config = {
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
        if self.teleport_position:
            update_controller_config(self, ik_controller_config)
            ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
            ik_ctrl.update_base_pose(self.robots[0].base_pos, self.robots[0].base_ori)
            pos = self.get_init_target_pos()
            set_robot_based_on_ee_pos(self, pos, self._eef_xquat, ik_ctrl)
            obs, reward, done, info = self._wrapped_env.step(np.zeros(7))
            self.num_steps += 100
        else:
            update_controller_config(self, ik_controller_config)
            update_controller_config(self, osc_controller_config)
            self.ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
            self.ik_ctrl.update_base_pose(
                self.robots[0].base_pos, self.robots[0].base_ori
            )

            self.osc_ctrl = controller_factory("OSC_POSE", osc_controller_config)
            self.osc_ctrl.update_base_pose(
                self.robots[0].base_pos, self.robots[0].base_ori
            )

            pos = self.get_init_target_pos()
            pos = np.concatenate((pos, self._eef_xquat))
            obs = mp_to_point(
                self,
                self.ik_ctrl,
                self.osc_ctrl,
                pos,
                grasp=False,
            )
            obs = self._flatten_obs(obs)
        self.ep_step_ctr = 0
        return obs

    def check_grasp(
        self,
    ):
        if self.name.endswith("Lift"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )
        elif self.name.endswith("PickPlaceCan"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.objects[self.object_id],
            )
        return is_grasped

    def get_target_pose(
        self,
    ):
        if self.name.endswith("Lift"):
            pose = np.array([0, 0, 0.05]) + self._eef_xpos
        elif self.name.endswith("PickPlaceCan"):
            pose = np.array(
                [
                    0.175,
                    0.3,
                    self.sim.data.body_xpos[self.obj_body_id[self.obj_to_use]][-1]
                    + 0.025,
                ]
            )
        return pose

    def step(self, action):
        o, r, d, i = self._wrapped_env.step(action)
        self.num_steps += 1
        self.ep_step_ctr += 1
        is_grasped = self.check_grasp()
        is_success = self._check_success()
        if self.ep_step_ctr == self.horizon and is_grasped and not is_success:
            target_pose = self.get_target_pose()
            if self.teleport_position:
                for _ in range(50):
                    self._wrapped_env.step(
                        np.concatenate((target_pose - self._eef_xpos, [0, 0, 0, 1]))
                    )
                    self.num_steps += 1
            else:
                mp_to_point(
                    self,
                    self.ik_ctrl,
                    self.osc_ctrl,
                    np.concatenate((target_pose, self._eef_xquat)),
                    grasp=True,
                    ignore_object_collision=True,
                )
            new_r = self.reward(action)
            if self.check_grasp() and new_r > r:
                r = new_r
        is_grasped = self.check_grasp()
        is_success = self._check_success()
        i["success"] = float(is_grasped)
        i["grasped"] = float(is_success)
        i["num_steps"] = self.num_steps
        if not self.teleport_position:
            i["mp_init_mse"] = self.mp_init_mse
        return o, r, d, i


class RobosuiteEnv(ProxyEnv):
    def __init__(self, env):
        super().__init__(env)
        for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
            self.camera_segmentations,
        ):

            # Add cameras associated to our arrays
            cam_sensors, cam_sensor_names = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                cam_segs=cam_segs,
                modality="image",
            )
            self.cam_sensor = cam_sensors
        self.num_steps = 0

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        o, r, d, i = super().step(action)
        self.num_steps += 1
        i["success"] = float(self._check_success())
        i["grasped"] = float(
            self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )
        )
        i["num_steps"] = self.num_steps
        return o, r, d, i
