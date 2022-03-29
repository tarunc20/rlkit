
import numpy as np
from robosuite.controllers import controller_factory
from rlkit.envs.proxy_env import ProxyEnv

def set_robot_based_on_ee_pos(env, pos, ctrl):
    joint_pos = ctrl.inverse_kinematics(pos, env._eef_xquat)
    env.robots[0].set_robot_joint_positions(joint_pos)
    return np.linalg.norm(env._eef_xpos - pos)

class MPEnv(ProxyEnv):
    def reset(self, **kwargs):
        controller_config = {'type': 'IK_POSE', 'ik_pos_limit': 0.02, 'ik_ori_limit': 0.05, 'interpolation': None, 'ramp_ratio': 0.2}
        controller_config["robot_name"] = self._wrapped_env.robots[0].name
        controller_config["sim"] = self._wrapped_env.robots[0].sim
        controller_config["eef_name"] = self._wrapped_env.robots[0].gripper.important_sites["grip_site"]
        controller_config["eef_rot_offset"] = self._wrapped_env.robots[0].eef_rot_offset
        controller_config["joint_indexes"] = {
            "joints": self._wrapped_env.robots[0].joint_indexes,
            "qpos": self._wrapped_env.robots[0]._ref_joint_pos_indexes,
            "qvel": self._wrapped_env.robots[0]._ref_joint_vel_indexes,
        }
        controller_config["actuator_range"] = self._wrapped_env.robots[0].torque_limits
        controller_config["policy_freq"] = self._wrapped_env.robots[0].control_freq
        controller_config["ndim"] = len(self._wrapped_env.robots[0].robot_joints)
        o = self._wrapped_env.reset(**kwargs)
        self.ik_ctrl = controller_factory("IK_POSE", controller_config)
        self.ik_ctrl.update_base_pose(self._wrapped_env.robots[0].base_pos, self._wrapped_env.robots[0].base_ori)
        pos = self._wrapped_env.sim.data.body_xpos[self._wrapped_env.cube_body_id] + np.array([0, 0, .05])
        error = set_robot_based_on_ee_pos(self._wrapped_env, pos, self.ik_ctrl)
        obs, reward, done, info = self._wrapped_env.step(np.zeros(7))

        return obs

