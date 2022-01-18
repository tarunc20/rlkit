import argparse
import os
import random
import subprocess

import numpy as np
import torch

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from a2c_ppo_acktr.main import experiment

    experiment(variant)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        exp_prefix = "test" + args.exp_prefix
    else:
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm_kwargs=dict(
            entropy_coef=0.01,
            value_loss_coef=0.5,
            lr=3e-4,
            num_mini_batch=64,
            ppo_epoch=10,
            clip_param=0.2,
            eps=1e-5,
            max_grad_norm=0.5,
        ),
        rollout_kwargs=dict(
            use_gae=True,
            gamma=0.8,
            gae_lambda=0.95,
            use_proper_time_limits=True,
        ),
        env_kwargs=dict(
            config='configs/tasks/rearrange/pick.yaml',
            arm_controller='ArmRAPSAction',
            arm_type='ArmRAPSAction',
            grip_controller='MagicGraspAction',
            max_episode_steps=5,
            data_path='data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pick.json.gz',
            gym_obs_keys=('ee_pos', 'is_holding', 'obj_goal_pos_sensor', 'relative_resting_position', 'robot_head_depth'),
            action_scale=0.5,
            goto_pose_iterations=50,
        ),
        actor_kwargs=dict(recurrent=True, hidden_size=64, hidden_activation="tanh"),
        num_processes=16,
        num_env_steps=int(1e7),
        log_interval=10,
        eval_interval=10,
        env_suite="habitat",
        use_linear_lr_decay=False,
        use_raw_actions=False,
    )

    search_space = {
        "env_kwargs.ee_ctrl_lim":[
            .005,
        ],
        "env_kwargs.ee_ctrl_quat_lim":[
            0.015,
        ],
        "num_processes":[16],
        "actor_kwargs.recurrent":[False],
        "base_type":['MLPCNN'],
        "env_kwargs.data_path":[
            # 'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pick.json.gz',
            # 'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pick_andrew2.json.gz',
            'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pickanyreceptacle.json.gz',
            # 'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pickanyreceptacle_1000.json.gz',
            # 'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pickanyobject_1000.json.gz',
            # 'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pickeverything_1000.json.gz',
            # 'data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/train/train_counter_L_analysis_5000_500.json.gz',
        ],
        "env_name":["pick"]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['num_seeds'] = args.num_seeds
        variant['exp_id'] = exp_id
        variant['debug'] = args.debug
        global exp_prefix_
        exp_prefix_ = args.exp_prefix
        variant['num_steps'] = 2048 // variant['num_processes']
        if args.debug:
            variant['num_steps'] = 5
        variant['python_cmd'] = subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1]
        for _ in range(variant['num_seeds']):
            seed = random.randint(0, 100000)
            variant["seed"] = int(seed)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix_,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="none",
                python_cmd=variant['python_cmd'],
                seed=seed,
                exp_id=int(variant['exp_id']),
                skip_wait=False,
            )
