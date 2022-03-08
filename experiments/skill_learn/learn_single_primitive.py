from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    setup_sweep_and_launch_exp,
)
from rlkit.torch.model_based.dreamer.experiments.learn_single_primitive_experiment import (
    experiment,
)

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=1,
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=1,
            min_num_steps_before_training=100,
            max_path_length=100,
            batch_size=256,
        )
    else:
        algorithm_kwargs = dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=100,
            batch_size=256,
        )
    variant = dict(
        algorithm="TD3",
        version="normal",
        algorithm_kwargs=algorithm_kwargs,
        env_suite="metaworld",
        env_name="sweep-into-v2",
        env_kwargs=dict(
            use_image_obs=True,
            imwidth=64,
            imheight=64,
            reward_type="dense",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
                max_path_length=100,
            ),
            action_space_kwargs=dict(
                control_mode="end_effector",
                action_scale=1,
                camera_settings={
                    "distance": 0.38227044687537043,
                    "lookat": [0.21052547, 0.32329237, 0.587819],
                    "azimuth": 141.328125,
                    "elevation": -53.203125160653144,
                },
                collect_primitives_info=False,
                render_intermediate_obs_to_info=False,
                set_primitive_goals=True,
                low_level_reward_type="argument_achievement",
                primitive_to_model="move_delta_ee",
            ),
        ),
        policy_kwargs=dict(
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
            state_encoder_kwargs=dict(hidden_sizes=[64, 64], output_size=64),
            joint_processor_args=(),
            joint_processor_kwargs=dict(hidden_sizes=[512, 256]),
            image_dim=64 * 64 * 3,
            scale=1,
        ),
        replay_buffer_size=int(1e6),
        trainer_kwargs=dict(
            discount=0.99,
        ),
        max_sigma=0.1,
    )

    setup_sweep_and_launch_exp(lambda x: x, variant, experiment, args)
