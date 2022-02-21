import os

import torch.nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.envs import primitives_make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import StableBaselinesVecEnv
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import (
    ActionSpaceSamplePolicy,
    DreamerPolicy,
)
from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
    EpisodeReplayBuffer,
    EpisodeReplayBufferSkillLearn,
)
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
from rlkit.torch.model_based.dreamer.rollout_functions import vec_rollout_skill_learn
from rlkit.torch.model_based.dreamer.world_models import WorldModel
from rlkit.torch.model_based.vec_managers import Manager, VecManager


def test_build_manager():
    variant = {
        "actor_kwargs": {
            "discrete_continuous_dist": True,
            "dist": "tanh_normal_dreamer_v1",
            "init_std": 0.0,
            "min_std": 0.1,
            "num_layers": 4,
        },
        "algorithm": "MultiTaskMultiManagerRAPS",
        "algorithm_kwargs": {
            "batch_size": 417,
            "max_path_length": 5,
            "min_num_steps_before_training": 5,
            "num_epochs": 5,
            "num_eval_steps_per_epoch": 5,
            "num_expl_steps_per_train_loop": 5,
            "num_pretrain_steps": 1,
            "num_train_loops_per_epoch": 1,
            "num_trains_per_train_loop": 1,
        },
        "env_kwargs": {
            "action_space_kwargs": {
                "action_scale": 1,
                "camera_settings": {
                    "azimuth": 141.328125,
                    "distance": 0.38227044687537043,
                    "elevation": -53.203125160653144,
                    "lookat": [0.21052547, 0.32329237, 0.587819],
                },
                "collect_primitives_info": True,
                "control_mode": "primitives",
                "num_low_level_actions_per_primitive": 5,
                "render_intermediate_obs_to_info": True,
            },
            "imheight": 64,
            "imwidth": 64,
            "reward_type": "sparse",
            "usage_kwargs": {
                "max_path_length": 5,
                "unflatten_images": False,
                "use_dm_backend": True,
                "use_raw_action_wrappers": False,
            },
            "use_image_obs": True,
        },
        "env_names": ["assembly-v2"],
        "env_suite": "metaworld",
        "exp_id": "0",
        "exp_name": "02-20-test_2022_02_20_16_49_24_0000--s-42632",
        "exp_prefix": "02-20-test",
        "expl_amount": 0.3,
        "instance_type": "None",
        "low_level_action_dim": 9,
        "max_path_length": 5,
        "model_kwargs": {
            "deterministic_state_size": 200,
            "gru_layer_norm": True,
            "model_hidden_size": 400,
            "pred_discount_num_layers": 3,
            "reward_num_layers": 2,
            "rssm_hidden_size": 200,
            "std_act": "sigmoid2",
            "stochastic_state_size": 50,
            "use_prior_instead_of_posterior": False,
        },
        "num_eval_envs": 1,
        "num_expl_envs": 10,
        "num_low_level_actions_per_primitive": 5,
        "pass_render_kwargs": True,
        "primitive_model_batch_size": 10425,
        "primitive_model_kwargs": {
            "image_dim": 12288,
            "image_encoder_args": [],
            "image_encoder_kwargs": {
                "input_channels": 3,
                "input_height": 64,
                "input_width": 64,
                "kernel_sizes": [4, 4, 4, 4],
                "n_channels": [16, 32, 64, 128],
                "paddings": [0, 0, 0, 0],
                "strides": [2, 2, 2, 2],
            },
            "joint_processor_args": [],
            "joint_processor_kwargs": {
                "hidden_sizes": [512, 256],
                "input_size": 576,
                "output_size": 9,
            },
            "state_encoder_args": [],
            "state_encoder_kwargs": {"hidden_sizes": [64, 64], "output_size": 64},
        },
        "primitive_model_replay_buffer_kwargs": {
            "low_level_action_dim": 9,
            "max_path_length": 5,
            "max_replay_buffer_size": 115384,
            "num_low_level_actions_per_primitive": 5,
        },
        "primitive_model_trainer_kwargs": {"policy_lr": 0.0003},
        "replay_buffer_kwargs": {
            "max_path_length": 5,
            "max_replay_buffer_size": 500000,
        },
        "save_video": True,
        "seed": "42632",
        "trainer_kwargs": {
            "actor_entropy_loss_schedule": "1e-4",
            "actor_lr": 8e-05,
            "adam_eps": 1e-05,
            "detach_rewards": False,
            "discount": 0.8,
            "forward_kl": False,
            "free_nats": 1.0,
            "imagination_horizon": 5,
            "kl_loss_scale": 0.0,
            "lam": 0.95,
            "policy_gradient_loss_scale": 1.0,
            "pred_discount_loss_scale": 10.0,
            "reward_loss_scale": 2.0,
            "target_update_period": 100,
            "transition_loss_scale": 0.8,
            "use_pred_discount": True,
            "vf_lr": 8e-05,
            "world_model_lr": 0.0003,
        },
        "use_raw_actions": False,
        "version": "normal",
        "vf_kwargs": {"num_layers": 3},
    }

    num_low_level_actions_per_primitive = variant["num_low_level_actions_per_primitive"]
    low_level_action_dim = variant["low_level_action_dim"]

    variant["primitive_model_kwargs"]["joint_processor_kwargs"][
        "hidden_activation"
    ] = nn.ReLU
    variant["primitive_model_kwargs"]["state_encoder_kwargs"][
        "hidden_activation"
    ] = nn.ReLU
    variant["primitive_model_kwargs"]["image_encoder_kwargs"][
        "hidden_activation"
    ] = nn.ReLU
    variant["env_kwargs"]["action_space_kwargs"]["primitive_model_kwargs"] = variant[
        "primitive_model_kwargs"
    ]
    primitive_model_path = os.path.join("test", "primitive_model.ptc")
    variant["env_kwargs"]["action_space_kwargs"][
        "primitive_model_path"
    ] = primitive_model_path

    def make_manager():
        manager_idx = 0
        ptu.set_gpu_mode(True, gpu_id=manager_idx)
        os.environ["EGL_DEVICE_ID"] = str(manager_idx)
        env_suite = variant.get("env_suite", "kitchen")
        env_name = variant["env_names"][manager_idx]
        env_kwargs = variant["env_kwargs"]
        num_expl_envs = variant["num_expl_envs"]
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
            device_id=manager_idx,
            reload_state_args=(
                num_expl_envs,
                primitives_make_env.make_env,
                (env_suite, env_name, env_kwargs),
            ),
        )
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(1)
        ]
        eval_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
            device_id=manager_idx,
            reload_state_args=(
                num_expl_envs,
                primitives_make_env.make_env,
                (env_suite, env_name, env_kwargs),
            ),
        )
        num_primitives = eval_env.num_primitives
        discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
        continuous_action_dim = expl_env.max_arg_len
        discrete_action_dim = expl_env.num_primitives
        if not discrete_continuous_dist:
            continuous_action_dim = continuous_action_dim + discrete_action_dim
            discrete_action_dim = 0
        action_dim = continuous_action_dim + discrete_action_dim
        use_batch_length = False
        obs_dim = expl_env.observation_space.low.size

        world_model = WorldModel(
            action_dim,
            image_shape=expl_env.image_shape,
            **variant["model_kwargs"],
        )
        actor = ActorModel(
            variant["model_kwargs"]["model_hidden_size"],
            world_model.feature_size,
            hidden_activation=nn.ELU,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            **variant["actor_kwargs"],
        )
        vf = Mlp(
            hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
            * variant["vf_kwargs"]["num_layers"],
            output_size=1,
            input_size=world_model.feature_size,
            hidden_activation=nn.ELU,
        )
        target_vf = Mlp(
            hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
            * variant["vf_kwargs"]["num_layers"],
            output_size=1,
            input_size=world_model.feature_size,
            hidden_activation=nn.ELU,
        )

        expl_policy = DreamerPolicy(
            world_model,
            actor,
            obs_dim,
            action_dim,
            exploration=True,
            expl_amount=variant.get("expl_amount", 0.3),
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=discrete_continuous_dist,
        )
        eval_policy = DreamerPolicy(
            world_model,
            actor,
            obs_dim,
            action_dim,
            exploration=False,
            expl_amount=0.0,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=discrete_continuous_dist,
        )

        rand_policy = ActionSpaceSamplePolicy(expl_env)

        rollout_function_kwargs = dict(
            num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
            low_level_action_dim=low_level_action_dim,
            num_primitives=num_primitives,
        )

        expl_path_collector = VecMdpPathCollector(
            expl_env,
            expl_policy,
            save_env_in_snapshot=False,
            rollout_fn=vec_rollout_skill_learn,
            rollout_function_kwargs=rollout_function_kwargs,
        )

        eval_path_collector = VecMdpPathCollector(
            eval_env,
            eval_policy,
            save_env_in_snapshot=False,
            rollout_fn=vec_rollout_skill_learn,
            rollout_function_kwargs=rollout_function_kwargs,
        )

        variant["replay_buffer_kwargs"]["use_batch_length"] = use_batch_length
        replay_buffer = EpisodeReplayBuffer(
            num_expl_envs,
            obs_dim,
            action_dim,
            **variant["replay_buffer_kwargs"],
        )
        trainer = DreamerV2Trainer(
            actor,
            vf,
            target_vf,
            world_model,
            expl_env.image_shape,
            **variant["trainer_kwargs"],
        )
        trainer.to(ptu.device)

        return Manager(
            expl_env,
            eval_env,
            expl_path_collector,
            eval_path_collector,
            trainer,
            replay_buffer,
            pretrain_policy=rand_policy,
            **variant["algorithm_kwargs"],
        )

    make_manager()


def test_vec_manager():
    variant = {
        "actor_kwargs": {
            "discrete_continuous_dist": True,
            "dist": "tanh_normal_dreamer_v1",
            "init_std": 0.0,
            "min_std": 0.1,
            "num_layers": 4,
        },
        "algorithm": "MultiTaskMultiManagerRAPS",
        "algorithm_kwargs": {
            "batch_size": 417,
            "max_path_length": 5,
            "min_num_steps_before_training": 5,
            "num_epochs": 5,
            "num_eval_steps_per_epoch": 5,
            "num_expl_steps_per_train_loop": 5,
            "num_pretrain_steps": 1,
            "num_train_loops_per_epoch": 1,
            "num_trains_per_train_loop": 1,
        },
        "env_kwargs": {
            "action_space_kwargs": {
                "action_scale": 1,
                "camera_settings": {
                    "azimuth": 141.328125,
                    "distance": 0.38227044687537043,
                    "elevation": -53.203125160653144,
                    "lookat": [0.21052547, 0.32329237, 0.587819],
                },
                "collect_primitives_info": True,
                "control_mode": "primitives",
                "num_low_level_actions_per_primitive": 5,
                "render_intermediate_obs_to_info": True,
            },
            "imheight": 64,
            "imwidth": 64,
            "reward_type": "sparse",
            "usage_kwargs": {
                "max_path_length": 5,
                "unflatten_images": False,
                "use_dm_backend": True,
                "use_raw_action_wrappers": False,
            },
            "use_image_obs": True,
        },
        "env_names": ["assembly-v2"],
        "env_suite": "metaworld",
        "exp_id": "0",
        "exp_name": "02-20-test_2022_02_20_16_49_24_0000--s-42632",
        "exp_prefix": "02-20-test",
        "expl_amount": 0.3,
        "instance_type": "None",
        "low_level_action_dim": 9,
        "max_path_length": 5,
        "model_kwargs": {
            "deterministic_state_size": 200,
            "gru_layer_norm": True,
            "model_hidden_size": 400,
            "pred_discount_num_layers": 3,
            "reward_num_layers": 2,
            "rssm_hidden_size": 200,
            "std_act": "sigmoid2",
            "stochastic_state_size": 50,
            "use_prior_instead_of_posterior": False,
        },
        "num_eval_envs": 1,
        "num_expl_envs": 10,
        "num_low_level_actions_per_primitive": 5,
        "pass_render_kwargs": True,
        "primitive_model_batch_size": 10425,
        "primitive_model_kwargs": {
            "image_dim": 12288,
            "image_encoder_args": [],
            "image_encoder_kwargs": {
                "input_channels": 3,
                "input_height": 64,
                "input_width": 64,
                "kernel_sizes": [4, 4, 4, 4],
                "n_channels": [16, 32, 64, 128],
                "paddings": [0, 0, 0, 0],
                "strides": [2, 2, 2, 2],
            },
            "joint_processor_args": [],
            "joint_processor_kwargs": {
                "hidden_sizes": [512, 256],
                "input_size": 576,
                "output_size": 9,
            },
            "state_encoder_args": [],
            "state_encoder_kwargs": {"hidden_sizes": [64, 64], "output_size": 64},
        },
        "primitive_model_replay_buffer_kwargs": {
            "low_level_action_dim": 9,
            "max_path_length": 5,
            "max_replay_buffer_size": 115384,
            "num_low_level_actions_per_primitive": 5,
        },
        "primitive_model_trainer_kwargs": {"policy_lr": 0.0003},
        "replay_buffer_kwargs": {
            "max_path_length": 5,
            "max_replay_buffer_size": 500000,
        },
        "save_video": True,
        "seed": "42632",
        "trainer_kwargs": {
            "actor_entropy_loss_schedule": "1e-4",
            "actor_lr": 8e-05,
            "adam_eps": 1e-05,
            "detach_rewards": False,
            "discount": 0.8,
            "forward_kl": False,
            "free_nats": 1.0,
            "imagination_horizon": 5,
            "kl_loss_scale": 0.0,
            "lam": 0.95,
            "policy_gradient_loss_scale": 1.0,
            "pred_discount_loss_scale": 10.0,
            "reward_loss_scale": 2.0,
            "target_update_period": 100,
            "transition_loss_scale": 0.8,
            "use_pred_discount": True,
            "vf_lr": 8e-05,
            "world_model_lr": 0.0003,
        },
        "use_raw_actions": False,
        "version": "normal",
        "vf_kwargs": {"num_layers": 3},
    }

    num_low_level_actions_per_primitive = variant["num_low_level_actions_per_primitive"]
    low_level_action_dim = variant["low_level_action_dim"]

    variant["primitive_model_kwargs"]["joint_processor_kwargs"][
        "hidden_activation"
    ] = nn.ReLU
    variant["primitive_model_kwargs"]["state_encoder_kwargs"][
        "hidden_activation"
    ] = nn.ReLU
    variant["primitive_model_kwargs"]["image_encoder_kwargs"][
        "hidden_activation"
    ] = nn.ReLU
    variant["env_kwargs"]["action_space_kwargs"]["primitive_model_kwargs"] = variant[
        "primitive_model_kwargs"
    ]
    primitive_model_path = os.path.join("test", "primitive_model.ptc")
    variant["env_kwargs"]["action_space_kwargs"][
        "primitive_model_path"
    ] = primitive_model_path

    def make_manager(manager_idx):
        ptu.set_gpu_mode(True, gpu_id=manager_idx)
        os.environ["EGL_DEVICE_ID"] = str(manager_idx)
        env_suite = variant.get("env_suite", "kitchen")
        env_name = variant["env_names"][manager_idx]
        env_kwargs = variant["env_kwargs"]
        num_expl_envs = variant["num_expl_envs"]
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
            device_id=manager_idx,
            reload_state_args=(
                num_expl_envs,
                primitives_make_env.make_env,
                (env_suite, env_name, env_kwargs),
            ),
        )
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(1)
        ]
        eval_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
            device_id=manager_idx,
            reload_state_args=(
                num_expl_envs,
                primitives_make_env.make_env,
                (env_suite, env_name, env_kwargs),
            ),
        )
        num_primitives = eval_env.num_primitives
        discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
        continuous_action_dim = expl_env.max_arg_len
        discrete_action_dim = expl_env.num_primitives
        if not discrete_continuous_dist:
            continuous_action_dim = continuous_action_dim + discrete_action_dim
            discrete_action_dim = 0
        action_dim = continuous_action_dim + discrete_action_dim
        use_batch_length = False
        obs_dim = expl_env.observation_space.low.size

        world_model = WorldModel(
            action_dim,
            image_shape=expl_env.image_shape,
            **variant["model_kwargs"],
        )
        actor = ActorModel(
            variant["model_kwargs"]["model_hidden_size"],
            world_model.feature_size,
            hidden_activation=nn.ELU,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            **variant["actor_kwargs"],
        )
        vf = Mlp(
            hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
            * variant["vf_kwargs"]["num_layers"],
            output_size=1,
            input_size=world_model.feature_size,
            hidden_activation=nn.ELU,
        )
        target_vf = Mlp(
            hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
            * variant["vf_kwargs"]["num_layers"],
            output_size=1,
            input_size=world_model.feature_size,
            hidden_activation=nn.ELU,
        )

        expl_policy = DreamerPolicy(
            world_model,
            actor,
            obs_dim,
            action_dim,
            exploration=True,
            expl_amount=variant.get("expl_amount", 0.3),
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=discrete_continuous_dist,
        )
        eval_policy = DreamerPolicy(
            world_model,
            actor,
            obs_dim,
            action_dim,
            exploration=False,
            expl_amount=0.0,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=discrete_continuous_dist,
        )

        rand_policy = ActionSpaceSamplePolicy(expl_env)

        rollout_function_kwargs = dict(
            num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
            low_level_action_dim=low_level_action_dim,
            num_primitives=num_primitives,
        )

        expl_path_collector = VecMdpPathCollector(
            expl_env,
            expl_policy,
            save_env_in_snapshot=False,
            rollout_fn=vec_rollout_skill_learn,
            rollout_function_kwargs=rollout_function_kwargs,
        )

        eval_path_collector = VecMdpPathCollector(
            eval_env,
            eval_policy,
            save_env_in_snapshot=False,
            rollout_fn=vec_rollout_skill_learn,
            rollout_function_kwargs=rollout_function_kwargs,
        )

        variant["replay_buffer_kwargs"]["use_batch_length"] = use_batch_length
        replay_buffer = EpisodeReplayBuffer(
            num_expl_envs,
            obs_dim,
            action_dim,
            **variant["replay_buffer_kwargs"],
        )
        trainer = DreamerV2Trainer(
            actor,
            vf,
            target_vf,
            world_model,
            expl_env.image_shape,
            **variant["trainer_kwargs"],
        )
        trainer.to(ptu.device)

        return Manager(
            expl_env,
            eval_env,
            expl_path_collector,
            eval_path_collector,
            trainer,
            replay_buffer,
            pretrain_policy=rand_policy,
            **variant["algorithm_kwargs"],
        )

    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env_names"][0]
    env_kwargs = variant["env_kwargs"]
    env_fns = [
        lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
        for _ in range(1)
    ]
    eval_env = StableBaselinesVecEnv(
        env_fns=env_fns,
        start_method="fork",
        device_id=0,
        reload_state_args=(
            1,
            primitives_make_env.make_env,
            (env_suite, env_name, env_kwargs),
        ),
    )
    discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
    continuous_action_dim = eval_env.max_arg_len
    discrete_action_dim = eval_env.num_primitives
    if not discrete_continuous_dist:
        continuous_action_dim = continuous_action_dim + discrete_action_dim
        discrete_action_dim = 0
    action_dim = continuous_action_dim + discrete_action_dim
    obs_dim = eval_env.observation_space.low.size

    primitive_model_buffer = EpisodeReplayBufferSkillLearn(
        variant["num_expl_envs"],
        obs_dim,
        action_dim,
        **variant["primitive_model_replay_buffer_kwargs"],
    )
    num_managers = 1
    manager_fns = [lambda: make_manager(i) for i in range(num_managers)]
    vec_manager = VecManager(
        manager_fns,
        variant["env_names"],
        start_method="fork",
        primitive_model_buffer=primitive_model_buffer,
    )
    vec_manager.collect_init_expl_paths()
    assert primitive_model_buffer._size > 0
