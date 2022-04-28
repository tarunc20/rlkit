import time
import cv2
import numpy as np
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.mprl.mp_env import (
    MPEnv,
    check_robot_collision,
    set_robot_based_on_ee_pos,
    update_controller_config,
)
from robosuite.controllers import controller_factory
from robosuite.controllers.controller_factory import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import robosuite as suite

if __name__ == "__main__":
    environment_kwargs = {
        "control_freq": 20,
        "controller": "OSC_POSE",
        "env_name": "PickPlaceBread",
        "hard_reset": False,
        "ignore_done": True,
        "reward_scale": 1.0,
        "robots": "Panda",
    }
    mp_env_kwargs = {
        "vertical_displacement": 0.04,
        "teleport_position": False,
        "planning_time": 20,
        "mp_bounds_low": (-1.45, -1.25, 0.45),
        "mp_bounds_high": (0.45, 0.85, 2.25),
        # "mp_bounds_low": (-2, -2, -2),
        # "mp_bounds_high": (2, 2, 2.2),
    }
    controller = environment_kwargs.pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    env = suite.make(
        **environment_kwargs,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
    )
    env = MPEnv(
        NormalizedBoxEnv(GymWrapper(env)),
        **mp_env_kwargs,
    )
    env.reset()
    positions = []
    update_controller_config(env, env.ik_controller_config)
    ik_ctrl = controller_factory("IK_POSE", env.ik_controller_config)
    ik_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)
    avg_error = 0
    for _ in range(1000):
        pos = np.random.uniform(
            mp_env_kwargs["mp_bounds_low"], mp_env_kwargs["mp_bounds_high"]
        )
        env.step(np.array([0, 0, 0, 0, 0, 0, 1]))
        # if check_robot_collision(env, False):
        #     cv2.imwrite('/home/mdalal/research/mprl/rlkit/test.png', env.get_image())
        #     exit()
        # error = set_robot_based_on_ee_pos(
        #     env, pos, env._eef_xquat, ik_ctrl, env.sim.data.qpos, env.sim.data.qvel
        # )
        # positions.append(env._eef_xpos)
        # avg_error += error
    # positions = np.array(positions)
    # print(f"Min position: {np.amin(positions, axis=0)}")
    # print(f"Max position: {np.amax(positions, axis=0)}")
    # print(f"Average error: {avg_error / 5000}")
