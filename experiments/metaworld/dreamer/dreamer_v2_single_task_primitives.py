import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.kitchen_dreamer import experiment

if __name__ == "__main__":
    cam_settings_list = [
        {
            "distance": 0.3211473534266694,
            "lookat": [0.29015772, 0.63492059, 0.544268],
            "azimuth": 178.59375,
            "elevation": -60.46875041909516,
        },
        {
            "distance": 0.513599996134662,
            "lookat": [0.28850459, 0.56757972, 0.54530015],
            "azimuth": 179.296875,
            "elevation": -47.34375002793968,
        },
        {
            "distance": 0.513599996134662,
            "lookat": [0.28839241, 0.55843923, 0.70374719],
            "azimuth": 179.82421875,
            "elevation": -59.76562483236194,
        },
        {
            "distance": 0.37864894603997346,
            "lookat": [0.28839241, 0.55843923, 0.70374719],
            "azimuth": -180.0,
            "elevation": -64.68749995809048,
        },
        {
            "distance": 0.38227044687537043,
            "lookat": [0.21052547, 0.32329237, 0.587819],
            "azimuth": 141.328125,
            "elevation": -53.203125160653144,
        },
        {
            "distance":0.513599996134662,
            "lookat":[0.28850459, 0.56757972, 0.54530015],
            "azimuth": 178.9453125,
            "elevation": -60.00000040512532,
        },

    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=30,
            max_path_length=5,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=2500,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=5,
            batch_size=417,  # 417*6 = 2502
            num_expl_steps_per_train_loop=30 * 2,  # 5*(5+1) one trajectory per vec env
            num_train_loops_per_epoch=40 // 2,  # 1000//(5*5)
            num_trains_per_train_loop=10 * 2,  # 400//40
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(5e5),
        algorithm_kwargs=algorithm_kwargs,
        use_raw_actions=False,
        env_suite="metaworld",
        pass_render_kwargs=True,
        env_kwargs=dict(
            control_mode="primitives",
            use_combined_action_space=True,
            action_scale=1,
            max_path_length=5,
            reward_type="sparse",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=False,
            ),
            image_kwargs=dict(imwidth=64, imheight=64),
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
            embedding_size=1024,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
        ),
        trainer_kwargs=dict(
            use_amp=True,
            opt_level="O1",
            optimizer_class="apex_adam",
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
        num_expl_envs=5 * 2,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
    )

    search_space = {
        "env_name": [
            # solveable
            "basketball-v2",
            # "assembly-v2",
            # "disassemble-v2"
            # verified and medium
            # "soccer-v2",
            # "sweep-into-v2",
            # easy
            # "plate-slide-v2",
            # "drawer-close-v2",
            # "faucet-open-v2",
            # verified and unsolved:
            # "bin-picking-v2",
            # unverified and unsolved:
            # "stick-pull-v2",
        ],
        "env_kwargs.camera_settings":[
            cam_settings_list[0],
            cam_settings_list[1],
            cam_settings_list[2],
            cam_settings_list[3],
            cam_settings_list[4],
            cam_settings_list[5],
            ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = preprocess_variant(variant, args.debug)
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
            )
