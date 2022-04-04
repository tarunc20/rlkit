def video_func(algorithm, epoch):
    import os
    import pickle
    import numpy as np
    from rlkit.torch.model_based.dreamer.visualization import make_video
    from rlkit.core import logger
    import copy

    if epoch % 50 == 0 or epoch == -1:
        policy = algorithm.eval_data_collector._policy
        max_path_length = algorithm.max_path_length
        env = algorithm.eval_env
        num_rollouts = 5
        frames = []
        for _ in range(num_rollouts):
            policy.reset()
            o = env.reset()
            for path_length in range(max_path_length):
                a, agent_info = policy.get_action(o)
                o, r, d, i = env.step(copy.deepcopy(a))
                im = env.get_image()
                if len(frames) > path_length:
                    frames[path_length] = np.concatenate(
                        (frames[path_length], im), axis=1
                    )
                else:
                    frames.append(im)
                if d:
                    break
        logdir = logger.get_snapshot_dir()
        make_video(frames, logdir, epoch)
        print("saved video for epoch {}".format(epoch))
        pickle.dump(policy, open(os.path.join(logdir, f"policy_{epoch}.pkl"), "wb"))


def load_policy(path):
    import pickle

    return pickle.load(open(path, "rb"))


def experiment(variant):
    import robosuite as suite
    from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
    from robosuite.wrappers import GymWrapper

    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.mprl.mp_env import MPEnv, RobosuiteEnv
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.torch.networks.mlp import ConcatMlp
    from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
    from rlkit.torch.sac.sac import SACTrainer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    # Get environment configs for expl and eval envs and create the appropriate envs
    # suites[0] is expl and suites[1] is eval
    suites = []
    for env_config in (
        variant["expl_environment_kwargs"],
        variant["eval_environment_kwargs"],
    ):
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        # Create robosuite env and append to our list
        suites.append(
            suite.make(
                **env_config,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_object_obs=True,
                use_camera_obs=False,
                reward_shaping=True,
                controller_configs=controller_config,
                camera_names="frontview",
                camera_heights=256,
                camera_widths=256,
            )
        )
    # Create gym-compatible envs

    if variant.get("mprl", False):
        expl_env = MPEnv(NormalizedBoxEnv(GymWrapper(suites[0])))
        eval_env = MPEnv(NormalizedBoxEnv(GymWrapper(suites[1])))
    else:
        expl_env = RobosuiteEnv(NormalizedBoxEnv(GymWrapper(suites[0])))
        eval_env = RobosuiteEnv(NormalizedBoxEnv(GymWrapper(suites[1])))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
    )

    # Instantiate trainer with appropriate agent
    expl_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant["policy_kwargs"],
    )
    eval_policy = MakeDeterministic(expl_policy)
    trainer = SACTrainer(
        env=eval_env,
        policy=expl_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant["trainer_kwargs"],
    )

    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
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
        **variant["algorithm_kwargs"],
    )
    algorithm.to(ptu.device)
    video_func(algorithm, -1)
    algorithm.post_epoch_funcs.append(video_func)
    algorithm.train()
