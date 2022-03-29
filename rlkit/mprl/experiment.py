def experiment(variant):
    from rlkit.mprl.mp_env import MPEnv

    from rlkit.torch.networks.mlp import ConcatMlp
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
    from rlkit.torch.sac.sac import SACTrainer
    import robosuite as suite
    from robosuite.wrappers import GymWrapper

    from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

    # Get environment configs for expl and eval envs and create the appropriate envs
    # suites[0] is expl and suites[1] is eval
    suites = []
    for env_config in (variant["expl_environment_kwargs"], variant["eval_environment_kwargs"]):
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        # Create robosuite env and append to our list
        suites.append(suite.make(**env_config,
                                 has_renderer=False,
                                 has_offscreen_renderer=False,
                                 use_object_obs=True,
                                 use_camera_obs=False,
                                 reward_shaping=True,
                                 controller_configs=controller_config,
                                 ))
    # Create gym-compatible envs

    if variant.get('mprl', False):
        expl_env = MPEnv(NormalizedBoxEnv(GymWrapper(suites[0])))
        eval_env = MPEnv(NormalizedBoxEnv(GymWrapper(suites[1])))
    else:
        expl_env = NormalizedBoxEnv(GymWrapper(suites[0]))
        eval_env = NormalizedBoxEnv(GymWrapper(suites[1]))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    # Define references to variables that are agent-specific
    trainer = None
    eval_policy = None
    expl_policy = None

    # Instantiate trainer with appropriate agent
    expl_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs'],
    )
    eval_policy = MakeDeterministic(expl_policy)
    trainer = SACTrainer(
        env=eval_env,
        policy=expl_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    # Define algorithm
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
