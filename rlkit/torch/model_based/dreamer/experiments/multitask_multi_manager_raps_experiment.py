def experiment(variant):
    import os

    from rlkit.core import logger
    from rlkit.torch.model_based.dreamer.conv_networks import CNNMLP
    from rlkit.torch.model_based.dreamer.td3 import TD3Trainer
    from rlkit.torch.model_based.rl_algorithm import TorchMultiManagerBatchRLAlgorithm

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch.nn as nn

    import rlkit.envs.primitives_make_env as primitives_make_env
    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.wrappers.mujoco_vec_wrappers import StableBaselinesVecEnv
    from rlkit.torch.model_based.dreamer.actor_models import ActorModel
    from rlkit.torch.model_based.dreamer.bc_trainer import BCTrainer
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
    from rlkit.torch.model_based.dreamer.rollout_functions import (
        vec_rollout_skill_learn,
    )
    from rlkit.torch.model_based.dreamer.visualization import post_epoch_visualize_func
    from rlkit.torch.model_based.dreamer.world_models import WorldModel
    from rlkit.torch.model_based.vec_managers import Manager, VecManager

    num_managers = len(variant["env_names"])

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
    primitive_model_path = os.path.join(
        logger.get_snapshot_dir(), "primitive_model.ptc"
    )
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
            manager_idx=manager_idx,
            **variant["algorithm_kwargs"],
        )

    if num_managers == 1:
        manager_fns = [lambda: make_manager(0)]
    elif num_managers == 2:
        manager_fns = [lambda: make_manager(0), lambda: make_manager(1)]
    elif num_managers == 3:
        manager_fns = [
            lambda: make_manager(0),
            lambda: make_manager(1),
            lambda: make_manager(2),
        ]
    elif num_managers == 4:
        manager_fns = [
            lambda: make_manager(0),
            lambda: make_manager(1),
            lambda: make_manager(2),
            lambda: make_manager(3),
        ]

    vec_manager = VecManager(
        manager_fns,
        variant["env_names"],
        start_method="forkserver",
    )
    obs_dim, action_dim, action_space = vec_manager.get_obs_and_action_dims()
    primitive_model_buffer = EpisodeReplayBufferSkillLearn(
        variant["num_expl_envs"],
        obs_dim,
        action_dim,
        **variant["primitive_model_replay_buffer_kwargs"],
    )
    vec_manager.set_primitive_model_buffer(primitive_model_buffer)

    variant["primitive_model_kwargs"]["state_encoder_kwargs"]["input_size"] = (
        action_dim + 1
    )
    variant["primitive_model_kwargs"]["output_activation"] = nn.Tanh()

    primitive_model = CNNMLP(**variant["primitive_model_kwargs"]).to(ptu.device)
    target_primitive_model = CNNMLP(**variant["primitive_model_kwargs"]).to(ptu.device)

    primitive_model_pretrain_trainer = BCTrainer(
        primitive_model, **variant["primitive_model_pretrain_trainer_kwargs"]
    )

    qf_kwargs = variant["primitive_model_kwargs"].copy()
    qf_kwargs["state_encoder_kwargs"]["input_size"] += variant["low_level_action_dim"]
    qf_kwargs["joint_processor_kwargs"]["output_size"] = 1
    qf1 = CNNMLP(**qf_kwargs).to(ptu.device)
    qf2 = CNNMLP(**qf_kwargs).to(ptu.device)
    target_qf1 = CNNMLP(**qf_kwargs).to(ptu.device)
    target_qf2 = CNNMLP(**qf_kwargs).to(ptu.device)

    if variant.get("use_rl_to_train_primitive_model", False):
        primitive_model_trainer = TD3Trainer(
            policy=primitive_model,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_primitive_model,
            **variant["primitive_model_trainer_kwargs"],
        )
    else:
        primitive_model_trainer = primitive_model_pretrain_trainer

    algorithm = TorchMultiManagerBatchRLAlgorithm(
        vec_manager,
        primitive_model_pretrain_trainer=primitive_model_pretrain_trainer,
        primitive_model_trainer=primitive_model_trainer,
        primitive_model_buffer=primitive_model_buffer,
        primitive_model_path=primitive_model_path,
        **variant["algorithm_kwargs"],
        **variant["primitive_model_algorithm_kwargs"],
    )
    algorithm.low_level_primitives = False
    if variant.get("generate_video", False):
        post_epoch_visualize_func(algorithm, 0)
    else:
        if variant.get("save_video", False):
            algorithm.post_epoch_funcs.append(post_epoch_visualize_func)
        print("TRAINING")
        algorithm.train()
        if variant.get("save_video", False):
            post_epoch_visualize_func(algorithm, -1)
