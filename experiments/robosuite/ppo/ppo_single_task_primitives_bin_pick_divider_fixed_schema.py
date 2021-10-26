import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from a2c_ppo_acktr.main import experiment

    experiment(variant)


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
            lr=5e-3,
            num_mini_batch=10,
            ppo_epoch=10,
            clip_param=0.2,
            eps=1e-5,
            max_grad_norm=0.5,
        ),
        rollout_kwargs=dict(
            use_gae=True,
            gamma=0.99,
            gae_lambda=0.95,
            use_proper_time_limits=True,
        ),
        env_kwargs=dict(
            robots="Panda",
            gripper_types="PandaGripper",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_heights=64,
            camera_widths=64,
            controller_configs=config,
            horizon=3,
            control_freq=40,
            reward_shaping=True,
            use_cube_shift_left_reward=False,
            use_reaching_reward=False,
            use_grasping_reward=False,
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
                action_scale=0.3,
                max_path_length=3,
                camera_settings={
                    "distance": 1.161288187018284,
                    "lookat": [-0.0, 0.0, 0.22],
                    "azimuth": 180,
                    "elevation": -90,
                },
                workspace_low=(-0.17, -0.17, 0.95),
                workspace_high=(0.17, 0.17, 0.99),
                reward_type="dense",
                fixed_schema=True,
                spatial_actions=True,
                imwidth=256,
                imheight=256,
            ),
            usage_kwargs=dict(
                use_dm_backend=True,
                max_path_length=3,
                use_image_obs=True,
                unflatten_images=True,
            ),
            image_kwargs=dict(),
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=64, hidden_activation="tanh"),
        num_processes=4,
        num_env_steps=int(5e4),
        num_steps=50 // 4,
        log_interval=1,
        eval_interval=1,
        use_raw_actions=False,
        env_suite="robosuite",
        discrete_continuous_dist=False,
        use_linear_lr_decay=False,
    )

    search_space = {
        "env_name": [
            "BinDividerPick",
        ],
        "rollout_kwargs.gamma": [0.99],
        "algorithm_kwargs.lr": [3e-4, 1e-3],
        "algorithm_kwargs.num_mini_batch": [10, 64],
        "num_steps": [500 // 4, 100 / 4],
        "actor_kwargs.use_pretrained_vgg": [False],
        # "env_kwargs.placement_initializer_kwargs.x_range":[(0, .165), (-.165, 0), (0, 0)],
        # "env_kwargs.use_cube_shift_left_reward": [True, False],
        # "env_kwargs.use_reaching_reward": [True, False],
        # "env_kwargs.use_grasping_reward": [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id
            run_experiment(
                experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="none",
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
                skip_wait=False,
            )
