import random
import subprocess
import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.policy_gradient.vpg.experiments import base_vpg_experiment
from rlkit.torch.policy_gradient.utils.arguments import get_args

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1000,
            max_path_length=1000,
            num_expl_steps_per_train_loop=10000,
            num_trains_per_train_loop=5,
            num_train_loops_per_epoch=5,
            batch_size=128
        )
        exp_prefix = "debug:" + args.exp_prefix

    else:
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256
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
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3e-4,
            reward_scale=1
        )
    )

    search_space = {
        "env_id": [
            "walker_walk",
            "pendulum_swingup", 
            "cartpole_swingup_sparse", 
            "hopper_stand",
            "walker_run",
            "cheetah_run"
        ]
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant
    )

    num_exps_launched = 0

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 1000000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id

            run_experiment(
                base_vpg_experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=False,
                snapshot_mode="none",
                python_cmd=subprocess.check_output('which python', shell=True).decode('utf-8')[:-1],
                seed=seed,
                exp_id=exp_id,
                skip_wait=False
            )

            num_exps_launched += 1
    
    print("Num exps launched: ", num_exps_launched)

            
