import rlkit.torch.pytorch_util as ptu
from gym.envs.mujoco import HalfCheetahEnv
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
# from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks.mlp import ConcatMlp
from rlkit.torch.policy_gradient.vpg.vpg import VPGTrainer
from rlkit.torch.policy_gradient.rl_algorithm import TorchBatchRLAlgorithm



def experiment(variant):
    
    expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    layer_size = variant["layer_size"]

    policy = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[layer_size, layer_size]        
    )

    eval_path_collector = MdpPathCollector(
        eval_env,
        policy
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        policy
    )

    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env
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
        layer_size=256,
        replay_buffer_size=int(1e6),
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
