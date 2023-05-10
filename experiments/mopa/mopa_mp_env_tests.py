import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import mujoco_py
import sys
import os
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import cv2
from rlkit.envs.wrappers import NormalizedBoxEnv
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from mopa_rl.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
import torch

torch.set_float32_matmul_precision("medium")
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

def video_func(algorithm, epoch):
    import copy
    import os
    import pickle

    import numpy as np

    from rlkit.core import logger
    from rlkit.torch.model_based.dreamer.visualization import make_video
    from rlkit.core.batch_rl_algorithm import BatchModularRLAlgorithm

    if epoch % 10 == 0 or epoch == -1 and epoch != 0:
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
        for step in range(max_path_length):
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
        env = algorithm.trainer.env
        algorithm.trainer.env = None
        pickle.dump(
            algorithm.trainer, open(os.path.join(logdir, f"control_{epoch}.pkl"), "wb")
        )
        algorithm.trainer.env = env
        if isinstance(algorithm, BatchModularRLAlgorithm):
            algorithm.planner_trainer.env = None
            pickle.dump(
                algorithm.planner_trainer,
                open(os.path.join(logdir, f"planner_{epoch}.pkl"), "wb"),
            )
            algorithm.planner_trainer.env = env


def make_env(variant):
    if variant["env_name"] == "SawyerLift-v0":
        config = LIFT_CONFIG
    elif variant["env_name"] == "SawyerLiftObstacle-v0":
        config = LIFT_OBSTACLE_CONFIG
    elif variant["env_name"] == "SawyerAssemblyObstacle-v0":
        config = ASSEMBLY_OBSTACLE_CONFIG 
    env = gym.make(**config)
    ik_env = gym.make(**config)
    env = MoPAMPEnv(
        variant["env_name"],
        env,
        ik_env,
        config=config,
        plan_to_learned_goals=variant["plan_to_learned_goals"],
        num_ll_actions_per_hl_action=variant["num_ll_actions_per_hl_action"],
        teleport_on_grasp=variant["teleport_on_grasp"],
        vertical_displacement=variant["vertical_displacement"],
        planner_command_orientation=variant["planner_command_orientation"],
        horizon=variant["algorithm_kwargs"]["max_path_length"],
        ignore_done=variant["ignore_done"]
    )
    return env



def full_experiment(variant):
    import numpy as np
    import torch
    #np.random.seed(0)
    #torch.manual_seed(0) 
    num_expl_envs = variant["num_expl_envs"]
    env_fns = [lambda: make_env(variant) for _ in range(num_expl_envs)]
    expl_env = StableBaselinesVecEnv(
        env_fns=env_fns,
        start_method="fork",
    )

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

    import torch
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

    hierarchical_eval_policy = StepBasedSwitchingPolicy(
        planner_eval_policy,
        eval_policy,
        policy2_steps_per_policy1_step=variant.get(
            "num_ll_actions_per_hl_action"
        ),
        use_episode_breaks=False,  # eval should not use episode breaks
    )
    hierarchical_expl_policy = StepBasedSwitchingPolicy(
        planner_expl_policy,
        expl_policy,
        policy2_steps_per_policy1_step=variant.get(
            "num_ll_actions_per_hl_action"
        ),
        use_episode_breaks=variant.get("use_episode_breaks", False),
        only_keep_trajs_after_grasp_success=variant.get(
            "only_keep_trajs_after_grasp_success", False
        ),
        only_keep_trajs_stagewise=variant.get(
            "only_keep_trajs_stagewise", False
        ),
        terminate_each_stage=variant.get("terminate_each_stage", False),
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        hierarchical_eval_policy,
        rollout_fn=rollout_modular,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        hierarchical_expl_policy,
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
    # add video function 
    algorithm.to(ptu.device)
    algorithm.post_epoch_funcs.append(video_func)
    algorithm.train()

if __name__ == "__main__":
    from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
        setup_sweep_and_launch_exp,
    )
    import numpy as np

    from rlkit.mprl.experiment import experiment, preprocess_variant_mp
    from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
    from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
        setup_sweep_and_launch_exp,
    )
    def preprocess_variant(variant, **kwargs):
        return variant
    args = get_args()
    variant = dict(
        env_name="SawyerLift-v0",
        policy_kwargs=dict(hidden_sizes=(256, 256)),
        qf_kwargs=dict(hidden_sizes=(256, 256)),
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=0.001,
            qf_lr=0.0005,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=5,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_size=int(5e6),
        algorithm_kwargs=dict(
            batch_size=128,
            min_num_steps_before_training=3300,
            num_epochs=5000,
            num_eval_steps_per_epoch=500, # used to be 2500
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            max_path_length=100,
        ),
        mprl=True,
        plan_to_learned_goals=False, # set false for now 
        num_ll_actions_per_hl_action=50,
        vertical_displacement=0.05,
        debug=True,
        wandb=False,
        project="mopa_test",
        root_dir=os.getcwd()[:-6],
        sheet_name="motion-planning-rl",
        num_hl_actions_total=1,
        ignore_done=True,
        seed=36251,
        num_expl_envs=10,
        teleport_on_grasp=False,
        planner_command_orientation=True,
        planner_trainer_kwargs=dict(
            discount=0.5,
            policy_lr=0.001,
            qf_lr=0.0005,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=5,
            use_automatic_entropy_tuning=True,
        ),
    ) 
    setup_sweep_and_launch_exp(preprocess_variant, variant, full_experiment, args)
    