import os
import pickle

from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy


def preprocess_variant(variant, debug):
    variant["sheet_name"] = "motion-planning-rl"
    variant["project"] = "mprl"
    variant["wandb"] = True
    variant["root_dir"] = os.getcwd()[:-6]  # remove /rlkit
    variant["algorithm_kwargs"]["max_path_length"] = variant["max_path_length"]
    variant["algorithm_kwargs"]["num_eval_steps_per_epoch"] = (
        50 * variant["max_path_length"]
    )
    controller_configs = variant.pop("controller_configs")
    environment_kwargs = variant.pop("environment_kwargs")

    environment_kwargs["controller_configs"] = controller_configs
    environment_kwargs["horizon"] = variant["max_path_length"]

    environment_kwargs.update(variant.get("additional_reward_configs", {}))

    variant["expl_environment_kwargs"] = environment_kwargs
    variant["eval_environment_kwargs"] = environment_kwargs
    if "mp_env_kwargs" in variant:
        variant["mp_env_kwargs"]["controller_configs"] = controller_configs
    if debug:
        variant["debug"] = True
        algorithm_kwargs = variant["algorithm_kwargs"]
        algorithm_kwargs["min_num_steps_before_training"] = max(
            variant["max_path_length"], algorithm_kwargs["batch_size"]
        )
        algorithm_kwargs["num_epochs"] = 2
        algorithm_kwargs["num_eval_steps_per_epoch"] = variant["max_path_length"]
        algorithm_kwargs["num_expl_steps_per_train_loop"] = variant["max_path_length"]
        algorithm_kwargs["num_trains_per_train_loop"] = 1
        algorithm_kwargs["num_train_loops_per_epoch"] = 1
    return variant


def preprocess_variant_mp(variant, debug):
    variant = preprocess_variant(variant, debug)
    variant["mp_env_kwargs"]["plan_to_learned_goals"] = variant["plan_to_learned_goals"]
    if variant["plan_to_learned_goals"]:
        variant["algorithm_kwargs"]["planner_num_trains_per_train_loop"] = variant[
            "planner_num_trains_per_train_loop"
        ]
    return variant


# Create gym-compatible envs
def make_env(variant):
    from robosuite.wrappers import GymWrapper
    import robosuite as suite
    import gym
    from rlkit.mprl.mp_env import MPEnv, RobosuiteEnv
    from rlkit.envs.wrappers import NormalizedBoxEnv

    gym.logger.set_level(40)  # resolves annoying bbox warning
    expl_env = suite.make(
        **variant["expl_environment_kwargs"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=1024,
        camera_widths=1024,
    )
    if variant.get("mprl", False):
        expl_env = MPEnv(
            GymWrapper(expl_env),
            **variant.get("mp_env_kwargs"),
        )
    else:
        expl_env = RobosuiteEnv(
            NormalizedBoxEnv(GymWrapper(expl_env)),
            **variant.get("robosuite_env_kwargs"),
        )
    return expl_env


def video_func(algorithm, epoch):
    import copy
    import os
    import pickle

    import numpy as np

    from rlkit.core import logger
    from rlkit.torch.model_based.dreamer.visualization import make_video

    if epoch % 50 == 0 or epoch == -1 and epoch != 0:
        eval_collector = algorithm.eval_data_collector
        eval_env = algorithm.eval_env
        policy = eval_collector._policy
        max_path_length = algorithm.max_path_length
        num_envs = eval_env.num_envs
        frames = [[] for _ in range(num_envs)]
        success_rate = 0
        policy.reset()
        obs = eval_env.reset()
        images = eval_env.env_method("get_image")
        for i in range(num_envs):
            frames[i].append(images[i])
        for _ in range(max_path_length):
            actions, _ = policy.get_action(obs)
            obs, rewards, dones, infos = eval_env.step(copy.deepcopy(actions))
            images = eval_env.env_method("get_image")
            for i in range(num_envs):
                frames[i].append(images[i])
            if all(dones):
                break
        print(
            f"r: {rewards}, is grasped: {eval_env.env_method('check_grasp')}, logged grasp: {infos['grasped']}"
        )
        success_rate = eval_env.env_method("_check_success")
        logdir = logger.get_snapshot_dir()
        flattened_frames = []
        for i in range(num_envs):
            for frame in frames[i]:
                flattened_frames.append(frame)
        make_video(flattened_frames, logdir, epoch)
        print(f"Saved video for epoch {epoch}")
        print(f"Success rate: {np.mean(success_rate)}")
        pickle.dump(policy, open(os.path.join(logdir, f"policy_{epoch}.pkl"), "wb"))


def mp_video_func(algorithm, epoch):
    import copy
    import os
    import pickle

    import numpy as np

    from rlkit.core import logger
    from rlkit.torch.model_based.dreamer.visualization import add_text, make_video

    if epoch % 50 == 0 or epoch == -1 and epoch != 0:
        policy = algorithm.eval_data_collector._policy
        max_path_length = algorithm.max_path_length
        env = algorithm.eval_env.envs[0]
        num_rollouts = 5
        intermediate_frames_length = 100
        frames = []
        for _ in range(num_rollouts):
            policy.reset()
            o = env.reset(get_intermediate_frames=True)
            im = env.get_image()
            intermediate_frames = np.array(env.intermediate_frames)
            idxs = np.linspace(
                0, len(intermediate_frames), intermediate_frames_length, endpoint=False
            )
            if len(intermediate_frames) > 0:
                intermediate_frames = intermediate_frames[idxs.astype(int)]
            else:
                intermediate_frames = [im] * intermediate_frames_length
            if len(frames) > 0:
                for j, fr in enumerate(intermediate_frames):
                    frames[j] = np.concatenate([frames[j], fr], axis=1)
                frames[len(intermediate_frames)] = np.concatenate(
                    (frames[len(intermediate_frames)], im), axis=1
                )
                # frames[0] = np.concatenate((frames[0], im), axis=1)
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
                    intermediate_frames = np.array(env.intermediate_frames)
                    idxs = np.linspace(
                        0,
                        len(intermediate_frames),
                        intermediate_frames_length,
                        endpoint=False,
                    )
                    if len(intermediate_frames) > 0:
                        intermediate_frames = intermediate_frames[idxs.astype(int)]
                    else:
                        intermediate_frames = [im] * intermediate_frames_length
                    if len(frames) > path_length + prev_intermediate_frames_len:
                        for j, fr in enumerate(intermediate_frames):
                            frames[
                                j
                                + path_length
                                + prev_intermediate_frames_len
                                # j
                                # + prev_intermediate_frames_len
                            ] = np.concatenate(
                                [
                                    frames[
                                        j
                                        + path_length
                                        + prev_intermediate_frames_len
                                        # j
                                        # + prev_intermediate_frames_len
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
                        # frames[path_length] = np.concatenate(
                        #     (
                        #         frames[path_length],
                        #         im,
                        #     ),
                        #     axis=1,
                        # )
                    else:
                        frames.extend(intermediate_frames)
                        frames.append(im)
                else:
                    add_text(im, "Manipulator", (1, 10), 0.5, (0, 255, 0))
                    if len(frames) > path_length + len(intermediate_frames):
                        frames[path_length + len(intermediate_frames)] = np.concatenate(
                            (frames[path_length + len(intermediate_frames)], im), axis=1
                        )
                    else:
                        frames.append(im)
                if d:
                    break
            print(
                f"r:{r}, is grasped:{env.check_grasp()}, logged grasp: {i['grasped']}"
            )
            print(f"Success: {env._check_success()}")
        logdir = logger.get_snapshot_dir()
        make_video(frames, logdir, epoch)
        print("saved video for epoch {}".format(epoch))
        pickle.dump(policy, open(os.path.join(logdir, f"policy_{epoch}.pkl"), "wb"))


def mp_video_func_v4(algorithm, epoch):
    import copy
    import os
    import pickle

    import numpy as np

    from rlkit.core import logger
    from rlkit.torch.model_based.dreamer.visualization import add_text, make_video

    if epoch % 50 == 0 or epoch == -1 and epoch != 0:
        planner, policy = algorithm.eval_data_collector._policy
        max_path_length = algorithm.max_path_length
        env = algorithm.eval_env.envs[0]
        num_rollouts = 5
        intermediate_frames_length = 100
        frames = []
        for _ in range(num_rollouts):
            policy.reset()
            planner.reset()
            o = env.reset()
            im = env.get_image()
            if len(frames) > 0:
                frames[0] = np.concatenate((frames[0], im), axis=1)
            else:
                frames.append(im)
            for path_length in range(1, max_path_length + 3):
                if path_length == 1 or path_length == max_path_length + 2:
                    get_intermediate_frames = True
                    a, agent_info = planner.get_action(o)
                    if path_length == 1:
                        num_hl_steps = 0
                    else:
                        num_hl_steps = 1
                else:
                    get_intermediate_frames = False
                    a, agent_info = policy.get_action(o)
                o, r, d, i = env.step(
                    copy.deepcopy(a), get_intermediate_frames=get_intermediate_frames
                )
                im = env.get_image()
                add_text(im, "Manipulator", (1, 10), 0.5, (0, 255, 0))
                if get_intermediate_frames:
                    intermediate_frames = np.array(env.intermediate_frames)
                    idxs = np.linspace(
                        0,
                        len(intermediate_frames),
                        intermediate_frames_length,
                        endpoint=False,
                    )
                    if len(intermediate_frames) > 0:
                        intermediate_frames = intermediate_frames[idxs.astype(int)]
                    else:
                        intermediate_frames = [im] * intermediate_frames_length
                    if (
                        len(frames)
                        > path_length + num_hl_steps * intermediate_frames_length
                    ):
                        for j, fr in enumerate(intermediate_frames):
                            frames[
                                j
                                + path_length
                                + num_hl_steps * intermediate_frames_length
                            ] = np.concatenate(
                                [
                                    frames[
                                        j
                                        + path_length
                                        + num_hl_steps * intermediate_frames_length
                                    ],
                                    fr,
                                ],
                                axis=1,
                            )
                        frames[
                            path_length
                            + num_hl_steps * intermediate_frames_length
                            + len(intermediate_frames)
                        ] = np.concatenate(
                            (
                                frames[
                                    num_hl_steps * intermediate_frames_length
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
                    add_text(im, "Manipulator", (1, 10), 0.5, (0, 255, 0))
                    if len(frames) > path_length + len(intermediate_frames):
                        frames[path_length + len(intermediate_frames)] = np.concatenate(
                            (frames[path_length + len(intermediate_frames)], im), axis=1
                        )
                    else:
                        frames.append(im)
                if d:
                    break
            print(
                f"r:{r}, is grasped:{env.check_grasp()}, logged grasp: {i['grasped']}"
            )
            print(f"Success: {env._check_success()}")
        logdir = logger.get_snapshot_dir()
        make_video(frames, logdir, epoch)
        print("saved video for epoch {}".format(epoch))
        pickle.dump(policy, open(os.path.join(logdir, f"policy_{epoch}.pkl"), "wb"))


def load_policy(path):
    import pickle

    return pickle.load(open(path, "rb"))


def experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.envs.wrappers.mujoco_vec_wrappers import (
        DummyVecEnv,
        StableBaselinesVecEnv,
    )
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.samplers.rollout_functions import rollout_modular, vec_rollout
    from rlkit.torch.networks.mlp import ConcatMlp
    from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
    from rlkit.torch.sac.sac import SACTrainer
    from rlkit.torch.torch_rl_algorithm import (
        TorchBatchModularRLAlgorithm,
        TorchBatchRLAlgorithm,
    )

    num_expl_envs = variant.get("num_expl_envs", 1)
    if num_expl_envs > 1:
        env_fns = [lambda: make_env(variant) for _ in range(num_expl_envs)]
        expl_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
        )
    else:
        expl_envs = [make_env(variant)]
        expl_env = DummyVecEnv(expl_envs, pass_render_kwargs=False)

    eval_env = expl_env

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
            StepBasedSwitchingPolicy(
                planner_eval_policy,
                eval_policy,
                policy2_path_length=variant.get("max_path_length"),
            ),
            rollout_fn=rollout_modular,
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            StepBasedSwitchingPolicy(
                planner_expl_policy,
                expl_policy,
                policy2_path_length=variant.get("max_path_length"),
            ),
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
    algorithm.to(ptu.device)
    if variant.get("load_path", None):
        policy = pickle.load(open(os.path.join(variant["load_path"]), "rb"))
        policy = MakeDeterministic(policy)
        algorithm.eval_data_collector._policy = policy
        if variant.get("mp_env_kwargs", None):
            if variant["mp_env_kwargs"]["teleport_position"]:
                func = video_func
            else:
                if variant["plan_to_learned_goals"]:
                    func = mp_video_func_v4
                else:
                    func = mp_video_func
            func(algorithm, -1)
    else:
        if variant.get("mp_env_kwargs", None):
            if variant["mp_env_kwargs"]["teleport_position"]:
                func = video_func
            else:
                if variant["plan_to_learned_goals"]:
                    func = mp_video_func_v4
                else:
                    func = mp_video_func
        else:
            func = video_func
        func(algorithm, -1)
        algorithm.post_epoch_funcs.append(func)
        algorithm.train()
