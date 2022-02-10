from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    setup_sweep_and_launch_exp,
)
from rlkit.torch.policy_gradient.vpg.experiments.base_vpg_experiment import experiment


def preprocess_variant(variant):
    return variant


if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=0,
            max_path_length=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=128,
        )
        exp_prefix = "debug:" + args.exp_prefix

    else:
        algorithm_kwargs = dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        )

        exp_prefix = args.exp_prefix

    variant = dict(
        algorithm="VPG",
        version="normal",
        expl_env_id="HalfCheetah-v2",
        eval_env_id="HalfCheetah-v2",
        layer_size=256,
        replay_buffer_size=int(1e4),
        algorithm_kwargs=algorithm_kwargs,
        trainer_kwargs=dict(discount=0.99, policy_lr=3e-4, reward_scale=1),
    )

    search_space = {
        "env_id": [
            "walker_walk",
            "pendulum_swingup",
            "cartpole_swingup_sparse",
            "hopper_stand",
            "walker_run",
            "cheetah_run",
        ]
    }

    setup_sweep_and_launch_exp(preprocess_variant, variant, experiment, args)
