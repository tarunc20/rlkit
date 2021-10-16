import time

import cv2
import gym
import numpy as np
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

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
    # max_path_length = 5
    max_path_length = 200
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
            reward_shaping=False,
            reset_action_space_kwargs=dict(
                control_mode="primitives",
                action_scale=1,
                max_path_length=max_path_length,
                camera_settings={
                    "distance": 1.0029547420489309,
                    "lookat": np.array([-0.247022, 0.13217877, 0.41499862]),
                    "azimuth": 147.65625,
                    "elevation": -52.499999951105565,
                },
                workspace_low=(-0.15, -0.2, 0.8),
                workspace_high=(0.15, 0.2, 1),
            ),
            usage_kwargs=dict(
                use_dm_backend=True,
                max_path_length=max_path_length,
            ),
            image_kwargs=dict(),
        ),
    )
    o = env.reset()

    # cv2.imwrite('test.png', o.reshape(3, 64, 64).transpose(1, 2, 0))
    # while i in range(1000):
    # print(env.sim.data.body_xpos[env.cube_body_id])
    # print(env.sim.data.qpos)
    # a = np.array([
    #     -0.47844416,
    #     0.42349762,
    #     -0.28506953,
    #     0.05447144,
    #     -0.1029604,
    #     -0.7435894,
    #     0.97611815,
    # ])
    # import time
    # t = time.time()
    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "move_delta_ee_pose"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = [
        0.075,
        0.075,
        0,
    ]
    o, r, d, info = env.step(a, render_every_step=False, render_mode="human")

    a = np.zeros_like(a)
    primitive = "top_grasp"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.2
    o, r, d, info = env.step(a, render_every_step=False, render_mode="human")

    # a = np.zeros_like(a)
    # primitive = "close_gripper"
    # print(env._eef_xpos)
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")
    # print(env._eef_xpos)

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "lift"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.15
    o, r, d, info = env.step(a, render_every_step=False, render_mode="human")

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "move_left"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.25
    o, r, d, info = env.step(a, render_every_step=False, render_mode="human")

    a = env.action_space.sample()
    a = np.zeros_like(a)
    primitive = "open_gripper"
    a[env.get_idx_from_primitive_name(primitive)] = 1
    o, r, d, info = env.step(a, render_every_step=False, render_mode="human")

    print(r)

    # o, r, d, info = env.step(a, render_every_step=False, render_mode="human")
    # print(env._check_success(), r, info)
    # print(env._eef_xpos)
    # print(time.time()-t)
    # cv2.imwrite('test2.png', o.reshape(3, 64, 64).transpose(1, 2, 0))

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "move_left"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.5
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "move_backward"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.4
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "drop"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.4
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")
    # print(env._eef_xpos)
    # env.reset()

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "move_right"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.5
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "move_forward"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.4
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "lift"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.4
    # o, r, d, info = env.step(a, render_every_step=True, render_mode="human")
    # print(env._eef_xpos)
    # print(env._check_success(), r, info)
    # cv2.imwrite('test3.png', o.reshape(3, 64, 64).transpose(1, 2, 0))
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
