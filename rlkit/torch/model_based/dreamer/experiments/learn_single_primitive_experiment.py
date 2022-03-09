def experiment(variant):
    import torch.nn as nn

    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.simple_replay_buffer import ImageReplayBuffer
    from rlkit.envs.primitives_make_env import make_env
    from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.torch.model_based.dreamer.conv_networks import CNNMLP
    from rlkit.torch.model_based.dreamer.td3 import TD3Trainer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_kwargs = variant["env_kwargs"]
    env_name = variant["env_name"]
    eval_env = make_env(env_suite, env_name, env_kwargs)
    expl_env = make_env(env_suite, env_name, env_kwargs)
    action_dim = expl_env.action_space.low.size

    policy_kwargs = variant["policy_kwargs"]
    policy_kwargs["state_encoder_kwargs"]["input_size"] = expl_env.primitive_goal_dim
    policy_kwargs["joint_processor_kwargs"]["output_size"] = action_dim
    policy_kwargs["joint_processor_kwargs"]["input_size"] = (
        policy_kwargs["image_encoder_kwargs"]["n_channels"][-1] * 4
        + policy_kwargs["state_encoder_kwargs"]["output_size"]
    )
    policy_kwargs["output_activation"] = nn.Tanh()
    policy_kwargs["state_encoder_kwargs"]["hidden_activation"] = nn.ReLU
    policy_kwargs["state_encoder_kwargs"]["hidden_activation"] = nn.ReLU
    policy_kwargs["image_encoder_kwargs"]["output_activation"] = nn.ReLU
    policy_kwargs["joint_processor_kwargs"]["hidden_activation"] = nn.ReLU
    policy = CNNMLP(**policy_kwargs).to(ptu.device)
    target_policy = CNNMLP(**policy_kwargs).to(ptu.device)

    qf_kwargs = policy_kwargs.copy()
    qf_kwargs["state_encoder_kwargs"]["input_size"] += action_dim
    qf_kwargs["joint_processor_kwargs"]["output_size"] = 1
    qf_kwargs["output_activation"] = ptu.identity
    qf1 = CNNMLP(**qf_kwargs).to(ptu.device)
    qf2 = CNNMLP(**qf_kwargs).to(ptu.device)
    target_qf1 = CNNMLP(**qf_kwargs).to(ptu.device)
    target_qf2 = CNNMLP(**qf_kwargs).to(ptu.device)
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=variant["max_sigma"],
        min_sigma=variant["max_sigma"],  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = ImageReplayBuffer(
        variant["replay_buffer_size"],
        expl_env.observation_space.low.size + expl_env.primitive_goal_dim,
        action_dim,
        64 * 64 * 3,
        dict(),
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant["trainer_kwargs"]
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()
