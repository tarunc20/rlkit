import time

import cv2
import gym
import numpy as np
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import mat2euler

from rlkit.envs.primitives_make_env import make_env

if __name__ == "__main__":
    config = {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.2, 0.2, 0.2, 0.5, 0.5, 0.5],
        "output_min": [-0.2, -0.2, -0.2, -0.5, -0.5, -0.5],
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
    max_path_length = 5
    env = make_env(
        "robosuite",
        "BinDividerPick",
        dict(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_heights=64,
            camera_widths=64,
            controller_configs=config,
            horizon=max_path_length,
            control_freq=40,
            reward_shaping=True,
            use_cube_shift_left_reward=True,
            use_reaching_reward=True,
            use_grasping_reward=True,
            reset_action_space_kwargs=dict(
                control_mode="primitives",
                action_scale=1,
                max_path_length=max_path_length,
                camera_settings={
                    "distance": 1.161288187018284,
                    "lookat": np.array([-0.26495827, 0.07813156, 0.49040222]),
                    "azimuth": 159.43359375,
                    "elevation": -53.20312497206032,
                },
                workspace_low=(-0.17, -0.075, 0.95),
                workspace_high=(0.17, 0.17, 1.0),
                reward_type="dense",
            ),
            usage_kwargs=dict(
                use_dm_backend=True,
                max_path_length=max_path_length,
            ),
            image_kwargs=dict(),
        ),
    )
    o = env.reset()
    print(env.reward())
    import time

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "move_delta_ee_pose"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = [
        0.15,
        0.25,
        -0.03,
    ]
    o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    print(r)

    a = np.zeros_like(a)
    primitive = "close_gripper"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    print(r)

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "lift"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[
        env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])
    ] = 0.025
    o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    print(r)

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "move_left"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.25
    o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    print(r)

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "open_gripper"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    print(r)
    # for i in range(10000):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(
    #         action, render_every_step=True, render_mode="human"
    #     )  # take action in the environment
    #     # env.render(render_mode="rgb_array", imwidth=64, imheight=64)
    #     # cv2.imwrite('test/test_{}.png'.format(i), obs.reshape(3, 64, 64).transpose(1, 2, 0))
    #     # print(_, done)
    #     print(i)
    #     if done:
    #         env.reset()
