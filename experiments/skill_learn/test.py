import os

import cv2
import numpy as np
import torch.nn as nn

from rlkit.envs.primitives_make_env import make_env

if __name__ == "__main__":
    env_suite = "metaworld"
    env_name = "sweep-into-v2"

    primitive_model_kwargs = dict(
        image_encoder_args=(),
        image_encoder_kwargs=dict(
            input_width=64,
            input_height=64,
            input_channels=3,
            kernel_sizes=[4] * 4,
            n_channels=[16, 16 * 2, 16 * 4, 16 * 8],
            strides=[2] * 4,
            paddings=[0] * 4,
        ),
        state_encoder_args=(),
        state_encoder_kwargs=dict(hidden_sizes=[64, 64, 64], output_size=64),
        joint_processor_args=(),
        joint_processor_kwargs=dict(hidden_sizes=[512, 256], output_size=5),
        image_dim=64 * 64 * 3,
        scale=15,
    )

    env_kwargs = dict(
        use_image_obs=True,
        imwidth=64,
        imheight=64,
        reward_type="sparse",
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            unflatten_images=False,
            max_path_length=5,
        ),
        action_space_kwargs=dict(
            control_mode="primitives",
            action_scale=1,
            camera_settings={
                "distance": 0.38227044687537043,
                "lookat": [0.21052547, 0.32329237, 0.587819],
                "azimuth": 141.328125,
                "elevation": -53.203125160653144,
            },
            collect_primitives_info=True,
            render_intermediate_obs_to_info=True,
            low_level_reward_type="none",
            relabel_high_level_actions=True,
            num_low_level_actions_per_primitive=100,
            goto_pose_iterations=100,
        ),
    )

    primitive_model_kwargs["joint_processor_kwargs"]["input_size"] = (
        primitive_model_kwargs["image_encoder_kwargs"]["n_channels"][-1] * 4
        + primitive_model_kwargs["state_encoder_kwargs"]["output_size"]
    )

    primitive_model_kwargs["state_encoder_kwargs"]["hidden_activation"] = nn.ReLU
    primitive_model_kwargs["state_encoder_kwargs"]["hidden_activation"] = nn.ReLU
    primitive_model_kwargs["image_encoder_kwargs"]["output_activation"] = nn.ReLU
    primitive_model_kwargs["joint_processor_kwargs"]["hidden_activation"] = nn.ReLU
    primitive_model_kwargs["joint_processor_kwargs"]["output_activation"] = nn.Tanh()
    primitive_model_path = os.path.join(
        "data/03-19-save-trained-primitive-model/03-19-save_trained_primitive_model_2022_03_19_11_31_56_0000--s-15544/",
        "primitive_model.ptc",
    )
    env_kwargs["action_space_kwargs"]["primitive_model_path"] = primitive_model_path

    env_kwargs["action_space_kwargs"]["primitive_model_kwargs"] = primitive_model_kwargs

    env = make_env(
        env_suite,
        env_name,
        env_kwargs,
    )
    env.sync_primitive_model()
    env.set_use_primitive_model()
    o = env.reset()
    cv2.imwrite("test0.png", o.reshape(3, 64, 64).transpose(1, 2, 0))

    a = env.action_space.sample()
    a[1] = 1
    a[env.num_primitives + env.primitive_name_to_action_idx["move_along_x"]] = -0.5
    o, r, d, i = env.step(a)
    cv2.imwrite("test1.png", o.reshape(3, 64, 64).transpose(1, 2, 0))

    # a = env.action_space.sample()
    # a[4] = 1
    # a[env.num_primitives + env.primitive_name_to_action_idx["move_gripper"]] = 1
    # o, r, d, i = env.step(a)
    # cv2.imwrite("test2.png", o.reshape(3, 64, 64).transpose(1, 2, 0))
