


from rlkit.samplers.rollout_functions import vec_rollout


def video_func(algorithm, epoch):

    import copy
    import os
    import pickle

    import numpy as np

    from rlkit.core import logger
    from rlkit.torch.model_based.dreamer.visualization import make_video

    if epoch % 50 == 0 or epoch == -1:
        policy = algorithm.eval_data_collector._policy
        max_path_length = algorithm.max_path_length
        env = algorithm.eval_env.envs[0]
        num_rollouts = 5
        frames = []
        for _ in range(num_rollouts):
            policy.reset()
            o = env.reset(get_intermediate_frames=True)
            im = env.get_image()
            intermediate_frames = env.intermediate_frames
            if len(frames) > 0:
                for j, fr in enumerate(intermediate_frames):
                    frames[j] = np.concatenate([frames[j], fr], axis=1)
                frames[len(intermediate_frames)] = np.concatenate(
                    (frames[len(intermediate_frames)], im), axis=1
                )
            else:
                frames.extend(intermediate_frames)
                frames.append(im)
            prev_intermediate_frames_len = len(intermediate_frames)
            for path_length in range(1, max_path_length + 1):
                a, agent_info = policy.get_action(o)
                if path_length == max_path_length:
                    get_intermediate_frames = True
                else:
                    get_intermediate_frames = False
                o, r, d, i = env.step(
                    copy.deepcopy(a), get_intermediate_frames=get_intermediate_frames
                )
                im = env.get_image()
                if get_intermediate_frames:
                    intermediate_frames = env.intermediate_frames
                    if len(frames) > path_length + prev_intermediate_frames_len:
                        for j, fr in enumerate(intermediate_frames):
                            frames[
                                j + path_length + prev_intermediate_frames_len
                            ] = np.concatenate(
                                [
                                    frames[
                                        j + path_length + prev_intermediate_frames_len
                                    ],
                                    fr,
                                ],
                                axis=1,
                            )
                        frames[
                            path_length
                            + prev_intermediate_frames_len
                            + len(intermediate_frames)
                        ] = np.concatenate(
                            (
                                frames[
                                    prev_intermediate_frames_len
                                    + len(intermediate_frames)
                                    + path_length
                                ],
                                im,
                            ),
                            axis=1,
                        )
                    else:
                        frames.extend(intermediate_frames)
                        frames.append(im)
                else:
                    if len(frames) > path_length + len(intermediate_frames):
                        frames[path_length + len(intermediate_frames)] = np.concatenate(
                            (frames[path_length + len(intermediate_frames)], im), axis=1
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
    from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv, StableBaselinesVecEnv
    from rlkit.samplers.rollout_functions import rollout_modular
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper

    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.mprl.mp_env import MPEnv, RobosuiteEnv
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.torch.networks.mlp import ConcatMlp
    from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
    from rlkit.torch.sac.sac import SACTrainer
    from rlkit.torch.torch_rl_algorithm import (
        TorchBatchModularRLAlgorithm,
        TorchBatchRLAlgorithm,
    )

    # Load controller
    controller = variant["expl_environment_kwargs"].pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    # Create robosuite env and append to our list

    controller = variant["eval_environment_kwargs"].pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    # Create gym-compatible envs
    def make_env_expl():
        expl_env = suite.make(
            **variant["expl_environment_kwargs"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            controller_configs=controller_config,
        )
        if variant.get("mprl", False):
            expl_env = MPEnv(
                NormalizedBoxEnv(GymWrapper(expl_env)),
                **variant.get("mp_env_kwargs"),
            )
        else:
            expl_env = RobosuiteEnv(NormalizedBoxEnv(GymWrapper(expl_env)))
        return expl_env

    def make_env_eval():
        eval_env = suite.make(
            **variant["eval_environment_kwargs"],
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
        if variant.get("mprl", False):
            eval_env = MPEnv(
                NormalizedBoxEnv(GymWrapper(eval_env)),
                **variant.get("mp_env_kwargs"),
            )
        else:
            eval_env = RobosuiteEnv(NormalizedBoxEnv(GymWrapper(eval_env)))
        return eval_env

    num_expl_envs = variant.get("num_expl_envs", 1)
    if num_expl_envs > 1:
        env_fns = [make_env_expl for _ in range(num_expl_envs)]
        expl_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
        )
    else:
        expl_envs = [make_env_expl()]
        expl_env = DummyVecEnv(expl_envs, pass_render_kwargs=False)

    eval_env = [make_env_eval()]
    eval_env = DummyVecEnv(eval_env, pass_render_kwargs=False)

    obs_dim = eval_env.observation_space.low.size
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

    if variant["plan_to_learned_goals"]:
        planner_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
        )
        planner_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
        )
        planner_target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
        )
        planner_target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"]
        )

        # Instantiate trainer with appropriate agent
        planner_expl_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant["policy_kwargs"],
        )
        planner_eval_policy = MakeDeterministic(planner_expl_policy)

        planner_trainer = SACTrainer(
            env=eval_env,
            policy=planner_expl_policy,
            qf1=planner_qf1,
            qf2=planner_qf2,
            target_qf1=planner_target_qf1,
            target_qf2=planner_target_qf2,
            **variant["planner_trainer_kwargs"],
        )

        planner_replay_buffer = EnvReplayBuffer(
            variant["replay_buffer_size"],
            expl_env,
        )

        eval_path_collector = MdpPathCollector(
            eval_env,
            (planner_eval_policy, eval_policy),
            rollout_fn=rollout_modular,
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            (planner_expl_policy, expl_policy),
            rollout_fn=rollout_modular,
        )

        # Define algorithm
        algorithm = TorchBatchModularRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            planner_replay_buffer=planner_replay_buffer,
            planner_trainer=planner_trainer,
            **variant["algorithm_kwargs"],
        )
    else:
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy,
            rollout_fn=vec_rollout,
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
            rollout_fn=vec_rollout,
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
    print(ptu.device)
    algorithm.to(ptu.device)
    if (
        not variant["mp_env_kwargs"]["teleport_position"]
        and not variant["plan_to_learned_goals"]
    ):
        video_func(algorithm, -1)
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.train()
