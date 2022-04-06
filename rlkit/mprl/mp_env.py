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


def set_robot_based_on_ee_pos(env, pos, ctrl):
    joint_pos = ctrl.inverse_kinematics(pos, env._eef_xquat)
    env.robots[0].set_robot_joint_positions(joint_pos)
    return np.linalg.norm(env._eef_xpos - pos)


def check_robot_string(string):
    return string.startswith("robot") or string.startswith("gripper")


def check_robot_collision(env):
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if check_robot_string(con1) or check_robot_string(con2):
            return True
    return False


def isCollisionFreeVertex(env, pos):
    set_robot_based_on_ee_pos(env, pos)
    return check_robot_collision(env)


class MPEnv(ProxyEnv):
    def __init__(self, env, mp_env, vertical_displacement, teleport_position=True):
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
        self.mp_env = mp_env
        self.vertical_displacement = vertical_displacement
        self.teleport_position = teleport_position

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def mp_to_point(self, pos, grasp=False):
        def isStateValid(state):
            pos = np.array([state.getX(), state.getY(), state.getZ()])
            set_robot_based_on_ee_pos(self.mp_env, pos, self.ik_ctrl)
            valid = not check_robot_collision(self.mp_env)
            return valid

        # TODO: set mp_env state to match that of the current env
        self.mp_env.sim.data.qpos[:] = self.sim.data.qpos
        self.mp_env.sim.data.qvel[:] = self.sim.data.qvel
        self.mp_env.sim.forward()

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
        start().setXYZ(*self.mp_env._eef_xpos)
        start().rotation().x = self.mp_env._eef_xquat[0]
        start().rotation().y = self.mp_env._eef_xquat[1]
        start().rotation().z = self.mp_env._eef_xquat[2]
        start().rotation().w = self.mp_env._eef_xquat[3]
        # create a random goal state
        goal = ob.State(space)
        goal().setXYZ(*(pos))
        goal().rotation().x = self.mp_env._eef_xquat[0]
        goal().rotation().y = self.mp_env._eef_xquat[1]
        goal().rotation().z = self.mp_env._eef_xquat[2]
        goal().rotation().w = self.mp_env._eef_xquat[3]
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
            for state in converted_path:
                state = np.array(state)
                desired_rot = quat2mat(state[3:])
                for _ in range(100):
                    current_state = np.concatenate([self._eef_xpos, self._eef_xquat])
                    current_rot = quat2mat(current_state[3:])
                    rot_delta = orientation_error(desired_rot, current_rot)
                    pos_delta = (state - current_state)[:3]
                    if grasp:
                        grip_ctrl = 1
                    else:
                        grip_ctrl = 0
                    action = np.concatenate((pos_delta, rot_delta, [grip_ctrl]))
                    if np.linalg.norm(action) < 1e-5:
                        break
                    obs = self.wrapped_env.step(action)[0]
                    self.num_steps += 1
        return obs

    def reset(self, **kwargs):
        self._wrapped_env.reset(**kwargs)
        controller_config = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        if self.teleport_position:
            controller_config["robot_name"] = self.robots[0].name
            controller_config["sim"] = self.robots[0].sim
            controller_config["eef_name"] = self.robots[0].gripper.important_sites[
                "grip_site"
            ]
            controller_config["eef_rot_offset"] = self.robots[0].eef_rot_offset
            controller_config["joint_indexes"] = {
                "joints": self.robots[0].joint_indexes,
                "qpos": self.robots[0]._ref_joint_pos_indexes,
                "qvel": self.robots[0]._ref_joint_vel_indexes,
            }
            controller_config["actuator_range"] = self.robots[0].torque_limits
            controller_config["policy_freq"] = self.robots[0].control_freq
            controller_config["ndim"] = len(self.robots[0].robot_joints)
            self.ik_ctrl = controller_factory("IK_POSE", controller_config)
            self.ik_ctrl.update_base_pose(
                self.robots[0].base_pos, self.robots[0].base_ori
            )
            pos = self.sim.data.body_xpos[self.cube_body_id] + np.array(
                [0, 0, self.vertical_displacement]
            )
            error = set_robot_based_on_ee_pos(self, pos, self.ik_ctrl)
            obs, reward, done, info = self._wrapped_env.step(np.zeros(7))
            self.num_steps += 100
        else:
            self.mp_env.reset()
            self.mp_env.sim.data.qpos[:] = self.sim.data.qpos
            self.mp_env.sim.data.qvel[:] = self.sim.data.qvel
            self.mp_env.sim.forward()

            controller_config["robot_name"] = self.mp_env.robots[0].name
            controller_config["sim"] = self.mp_env.robots[0].sim
            controller_config["eef_name"] = self.mp_env.robots[
                0
            ].gripper.important_sites["grip_site"]
            controller_config["eef_rot_offset"] = self.mp_env.robots[0].eef_rot_offset
            controller_config["joint_indexes"] = {
                "joints": self.mp_env.robots[0].joint_indexes,
                "qpos": self.mp_env.robots[0]._ref_joint_pos_indexes,
                "qvel": self.mp_env.robots[0]._ref_joint_vel_indexes,
            }
            controller_config["actuator_range"] = self.mp_env.robots[0].torque_limits
            controller_config["policy_freq"] = self.mp_env.robots[0].control_freq
            controller_config["ndim"] = len(self.mp_env.robots[0].robot_joints)
            self.ik_ctrl = controller_factory("IK_POSE", controller_config)
            self.ik_ctrl.update_base_pose(
                self.mp_env.robots[0].base_pos, self.mp_env.robots[0].base_ori
            )
            pos = self.mp_env.sim.data.body_xpos[self.mp_env.cube_body_id] + np.array(
                [0, 0, self.vertical_displacement]
            )
            obs = self.mp_to_point(pos)
        self.ep_step_ctr = 0
        return obs

    def step(self, action):
        o, r, d, i = self._wrapped_env.step(action)
        self.num_steps += 1
        self.ep_step_ctr += 1
        is_grasped = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=self.cube,
        )
        is_success = self._check_success()
        if self.ep_step_ctr == self.horizon and is_grasped and not is_success:
            action = np.array([0, 0, 0.05, 0, 0, 0, 1])
            # if self.teleport_position:
            for _ in range(50):
                self._wrapped_env.step(action)
                self.num_steps += 1
            # else:
            #     self.mp_to_point(self._eef_xpos + action[:3], grasp=True) #TODO: update this to handle the fact that the collisions should be checked between arm + obj. and env
            new_r = self.reward(action)
            if (
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=self.cube,
                )
                and new_r > r
            ):
                r = new_r
        is_grasped = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=self.cube,
        )
        is_success = self._check_success()
        i["success"] = float(is_grasped)
        i["grasped"] = float(is_success)
        i["num_steps"] = self.num_steps
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
