import gym
import pdb
from numpy import var
from rlkit.torch.policy_gradient.utils.episode_replay_buffer import EpisodeReplayBuffer
from rlkit.torch.policy_gradient.utils.path_collector import VectorizedMdpPathCollector
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
# from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks.mlp import ConcatMlp, MlpPolicy
from rlkit.torch.policy_gradient.vpg.vpg import VPGTrainer
from rlkit.torch.policy_gradient.rl_algorithm import TorchBatchRLAlgorithm
from rlkit.envs.wrappers.mujoco_vec_wrappers import StableBaselinesVecEnv
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy


def experiment(variant):
    expl_env_func = lambda:gym.make(variant["expl_env_id"])
    eval_env_func = lambda:gym.make(variant["eval_env_id"])
    
    expl_env = StableBaselinesVecEnv(env_fns=[expl_env_func], start_method="fork")
    eval_env = StableBaselinesVecEnv(env_fns=[eval_env_func], start_method="fork")
    
    expl_env = NormalizedBoxEnv(expl_env)
    eval_env = NormalizedBoxEnv(eval_env)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    layer_size = variant["layer_size"]

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[layer_size, layer_size]        
    )

    eval_policy = MakeDeterministic(policy)

    expl_path_collector = VectorizedMdpPathCollector(
        expl_env,
        policy
    )

    eval_path_collector = VectorizedMdpPathCollector(
        eval_env,
        eval_policy
    )

    replay_buffer = EpisodeReplayBuffer(
        max_replay_buffer_size=variant["replay_buffer_size"],
        env=expl_env,
        max_path_length=variant['algorithm_kwargs']['max_path_length'],
        replace=True,
        use_batch_length=False
    )

    trainer = VPGTrainer(
        env=eval_env,
        policy=policy,
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

if __name__ == '__main__':
    variant = dict(
        algorithm="VPG",
        version="normal",
        expl_env_id="HalfCheetah-v2",
        eval_env_id="HalfCheetah-v2",
        layer_size=256,
        replay_buffer_size=int(1e4),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256
        ),
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3e-4,
            reward_scale=1
        )
    )

    # setup_logger("Vanilla-Policy-Gradient", variant=variant)
    experiment(variant)
