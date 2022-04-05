import cv2
import numpy as np
from robosuite.controllers import controller_factory

from rlkit.envs.proxy_env import ProxyEnv


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

    def reach_point(self,):
        """

        """
        pass

    def reset(self, **kwargs):
        o = self._wrapped_env.reset(**kwargs)
        controller_config = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
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
        self.ik_ctrl.update_base_pose(self.robots[0].base_pos, self.robots[0].base_ori)
        pos = self.sim.data.body_xpos[self.cube_body_id] + np.array([0, 0, 0.025])
        error = set_robot_based_on_ee_pos(self, pos, self.ik_ctrl)
        obs, reward, done, info = self._wrapped_env.step(np.zeros(7))
        self.num_steps += 50  # assume it takes at least this many steps to actually reach near the cube (this might be unfair)
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
            for _ in range(50):
                self._wrapped_env.step(action)
                self.num_steps += 1
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
