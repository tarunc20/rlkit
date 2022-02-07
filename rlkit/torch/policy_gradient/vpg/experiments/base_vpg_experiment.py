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


def experiment(variant):
    env_func = lambda:gym.make(variant["env_id"])
    env = StableBaselinesVecEnv(env_fns=[env_func], start_method="fork")
    env = NormalizedBoxEnv(env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    layer_size = variant["layer_size"]

    policy = MlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[layer_size, layer_size]        
    )

    path_collector = VectorizedMdpPathCollector(
        env,
        policy
    )

    replay_buffer = EpisodeReplayBuffer(
        max_replay_buffer_size=variant["replay_buffer_size"],
        env=env,
        max_path_length=variant['algorithm_kwargs']['max_path_length'],
        replace=True,
        use_batch_length=False
    )

    trainer = VPGTrainer(
        env=env,
        policy=policy,
        **variant["trainer_kwargs"]
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        env=env,
        data_collector=path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )

    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == '__main__':
    variant = dict(
        algorithm="VPG",
        version="normal",
        env_id="HalfCheetah-v2",
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
