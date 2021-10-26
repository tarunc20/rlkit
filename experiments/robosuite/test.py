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
            camera_heights=256,
            camera_widths=256,
            controller_configs=config,
            horizon=max_path_length,
            control_freq=40,
            reward_shaping=True,
            use_cube_shift_left_reward=False,
            use_reaching_reward=False,
            use_grasping_reward=True,
            placement_initializer_kwargs=dict(
                name="ObjectSampler",
                x_range=[-0.165, 0.165],
                y_range=[0.035, 0.165],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.12,
            ),
            reset_action_space_kwargs=dict(
                control_mode="primitives",
                action_scale=1,
                max_path_length=max_path_length,
                camera_settings={
                    "distance": 1.161288187018284,
                    "lookat": np.array([-0.0, 0.0, 0.22]),
                    "azimuth": 180,
                    "elevation": -90,
                },
                workspace_low=(-0.17, -0.17, 0.95),
                workspace_high=(0.17, 0.17, 1.05),
                reward_type="dense",
                imwidth=256,
                imheight=256,
                go_to_pose_iterations=100,
            ),
            usage_kwargs=dict(
                use_dm_backend=True,
                max_path_length=max_path_length,
            ),
            image_kwargs=dict(),
        ),
    )
    np.set_printoptions(precision=3)
    render_every_step = False
    for i in range(100):
        o = env.reset()
        # print(env.reward())
        print(env.sim.data.body_xpos[env.cube_body_id])
        import time

        # a = env.action_space.sample()
        # a = np.zeros_like(a)
        # primitive = "lift"
        # a[env.get_idx_from_primitive_name(primitive)] = 1
        # a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 1
        # o, r, d, info = env.step(
        #     a, render_every_step=render_every_step, render_mode="human"
        # )
        # print(env.sim.data.qpos[:7])

        a = env.action_space.sample()
        a = np.zeros_like(a)
        primitive = "top_xy_grasp"
        a[env.get_idx_from_primitive_name(primitive)] = 1
        a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = [
            env.sim.data.body_xpos[env.cube_body_id][:2][0],
            env.sim.data.body_xpos[env.cube_body_id][:2][1],
            -0.1,
        ]
        o, r, d, info = env.step(
            a, render_every_step=render_every_step, render_mode="human"
        )

        # print(r)
        # cv2.imshow("test", o.reshape((3, 256, 256)).transpose(1, 2, 0))
        # cv2.waitKey(0)

        a = env.action_space.sample()
        a = np.zeros_like(a)
        primitive = "lift"
        a[env.get_idx_from_primitive_name(primitive)] = 1
        a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.3
        o, r, d, info = env.step(
            a, render_every_step=render_every_step, render_mode="human"
        )

        # print(r)

        # cv2.imshow("test", o.reshape((3, 256, 256)).transpose(1, 2, 0))
        # cv2.waitKey(0)

        a = env.action_space.sample()
        a = np.zeros_like(a)
        primitive = "move_left"
        a[env.get_idx_from_primitive_name(primitive)] = 1
        a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = 0.3
        o, r, d, info = env.step(
            a, render_every_step=render_every_step, render_mode="human"
        )

        print(r)
    # cv2.imshow("test", o.reshape((3, 256, 256)).transpose(1, 2, 0))
    # cv2.waitKey(0)

    # a = env.action_space.sample()
    # a = np.zeros_like(a)
    # primitive = "open_gripper"
    # a[env.get_idx_from_primitive_name(primitive)] = 1
    # o, r, d, info = env.step(
    #     a, render_every_step=render_every_step, render_mode="human"
    # )

    # print(r)
    # cv2.imshow("test", o.reshape((3, 256, 256)).transpose(1, 2, 0))
    # cv2.waitKey(0)
    # for i in range(10000):
    #     a = env.action_space.sample()
    #     a = np.zeros_like(a)
    #     primitive = "move_delta_ee_pose"
    #     a[env.get_idx_from_primitive_name(primitive)] = 1
    #     a[env.num_primitives + np.array(env.primitive_name_to_action_idx[primitive])] = [-1, -1,1 ]
    #     obs, reward, done, info = env.step(
    #         a, render_every_step=False, render_mode="human"
    #     )  # take action in the environment
    #     cv2.imshow("test", obs.reshape((3, 256, 256)).transpose(1, 2, 0))
    #     print(env._eef_xpos)
    #     cv2.waitKey(0)
    #     print(i)
    #     if done:
    #         env.reset()



