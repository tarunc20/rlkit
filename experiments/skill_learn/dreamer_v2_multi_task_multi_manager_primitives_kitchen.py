from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant_multi_task_multi_manager_raps,
    setup_sweep_and_launch_exp,
)
from rlkit.torch.model_based.dreamer.experiments.multitask_multi_manager_raps_experiment import (
    experiment,
)

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=5,
            num_expl_steps_per_train_loop=5,
            min_num_steps_before_training=5,
            num_pretrain_steps=1,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1,
            batch_size=417,
        )
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=60,
            min_num_steps_before_training=2400,
            num_pretrain_steps=100,
            batch_size=417,
            num_expl_steps_per_train_loop=60,
            num_train_loops_per_epoch=20,
            num_trains_per_train_loop=20,
        )
    variant = dict(
        algorithm="MultiTaskMultiManagerRAPS",
        version="normal",
        algorithm_kwargs=algorithm_kwargs,
        use_raw_actions=False,
        env_suite="kitchen",
        pass_render_kwargs=True,
        env_names=(
            "hinge_cabinet",
            "top_left_burner",
            "light_switch",
            "microwave",
        ),
        env_kwargs=dict(
            reward_type="sparse",
            use_image_obs=True,
            action_scale=1.4,
            use_workspace_limits=True,
            control_mode="primitives",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
            ),
            action_space_kwargs=dict(),
            collect_primitives_info=True,
            render_intermediate_obs_to_info=True,
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=50,
            deterministic_state_size=200,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
            use_prior_instead_of_posterior=False,
        ),
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.8,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
            imagination_horizon=5,
        ),
        replay_buffer_kwargs=dict(
            max_replay_buffer_size=int(5e5),
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
        max_path_length=5,
        num_low_level_actions_per_primitive=5,
        low_level_action_dim=9,
        primitive_model_kwargs=dict(
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
        ),
        primitive_model_replay_buffer_kwargs=dict(),
        primitive_model_trainer_kwargs=dict(
            policy_lr=3e-4,
        ),
        primitive_model_batch_size=417,
    )

    setup_sweep_and_launch_exp(
        preprocess_variant_multi_task_multi_manager_raps, variant, experiment, args
    )
