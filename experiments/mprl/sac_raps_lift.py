from rlkit.mprl.primitives_experiment import experiment, preprocess_variant
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    setup_sweep_and_launch_exp,
)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = get_args()
    variant = {
        "mprl": True,
        "use_raps": True,
        "algorithm": "MPRL-SAC",
        "max_path_length": 50,
        "algorithm_kwargs": {
            "batch_size": 128,
            "min_num_steps_before_training": 3300,
            "num_epochs": 5000,
            "num_eval_steps_per_epoch": 250,
            "num_expl_steps_per_train_loop": 1000,
            "num_trains_per_train_loop": 1000,
        },
        "eval_environment_kwargs": {
            "control_freq": 20,
            "controller": "OSC_POSE",
            "env_name": "Lift",
            "hard_reset": False,
            "ignore_done": True,
            "reward_scale": 1.0,
            "robots": "Panda",
            "usage_kwargs": dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
            ),
        },
        "expl_environment_kwargs": {
            "control_freq": 20,
            "controller": "OSC_POSE",
            "env_name": "Lift",
            "hard_reset": False,
            "ignore_done": True,
            "reward_scale": 1.0,
            "robots": "Panda",
            "usage_kwargs": dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
            ),
        },
        "mp_env_kwargs": {
            "vertical_displacement": 0.03,
            "teleport_position": False,
            "planning_time": 10,
            "mp_bounds_low": (-2, -2, -2),
            "mp_bounds_high": (2, 2, 2.2),
            "usage_kwargs": dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
            ),
        },
        "policy_kwargs": {"hidden_sizes": (256, 256)},
        "qf_kwargs": {"hidden_sizes": (256, 256)},
        "replay_buffer_size": int(1e6),
        "seed": 129,
        "trainer_kwargs": {
            "discount": 0.99,
            "policy_lr": 0.001,
            "qf_lr": 0.0005,
            "reward_scale": 1.0,
            "soft_target_tau": 0.005,
            "target_update_period": 5,
            "use_automatic_entropy_tuning": True,
        },
        "planner_trainer_kwargs": {
            "discount": 0.5,
            "policy_lr": 0.001,
            "qf_lr": 0.0005,
            "reward_scale": 1.0,
            "soft_target_tau": 0.005,
            "target_update_period": 5,
            "use_automatic_entropy_tuning": True,
        },
        "version": "normal",
        "plan_to_learned_goals": True,
        "num_expl_envs": 1,
        "planner_num_trains_per_train_loop": 1000,
    }
    setup_sweep_and_launch_exp(preprocess_variant, variant, experiment, args)
