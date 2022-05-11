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
        "horizon": 50,
    }
    mp_env_kwargs = {
        "vertical_displacement": 0.04,
        "teleport_position": False,
        "planning_time": .3,
        "mp_bounds_low": (-1.45, -1.25, 0.45),
        "mp_bounds_high": (0.45, 0.85, 2.25),
        "update_with_true_state": True,
        "grip_ctrl_scale": .0025,
        "plan_to_learned_goals": True,
        "clamp_actions": True,
        "backtrack_movement_fraction": 0.1,
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
    for i in range(1):
        env.reset()
        for _ in range(52):
            env.step(env.action_space.sample())
