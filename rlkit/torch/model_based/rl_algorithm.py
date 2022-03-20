import abc
import os
import pickle
import time
from collections import OrderedDict

import gtimer as gt
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core import eval_util, logger
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector, PathCollector
from rlkit.torch.model_based.dreamer.utils import lambda_return_np


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times[f"time/{key} (s)"] = time
    times["time/epoch (s)"] = epoch_time
    times["time/total (s)"] = gt.get_times().total
    return times


class BaseRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: DataCollector,
        evaluation_data_collector: DataCollector,
        replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError("_train must implemented by inherited class")

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp("saving")
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for key, value in self.trainer.get_snapshot().items():
            snapshot["trainer/" + key] = value
        for key, value in self.expl_data_collector.get_snapshot().items():
            snapshot["exploration/" + key] = value
        for key, value in self.eval_data_collector.get_snapshot().items():
            snapshot["evaluation/" + key] = value
        for key, value in self.replay_buffer.get_snapshot().items():
            snapshot["replay_buffer/" + key] = value
        return snapshot

    def _log_stats(self, epoch):
        logger.log(f"Epoch {epoch} finished", with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(), prefix="replay_buffer/"
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix="trainer/")

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(), prefix="exploration/"
        )

        expl_paths = self.expl_data_collector.get_epoch_paths()
        if len(expl_paths) > 0:
            if hasattr(self.expl_env, "get_diagnostics"):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix="exploration/",
                )

            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix="evaluation/",
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, "get_diagnostics"):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix="evaluation/",
            )

        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp("logging")
        timings = _get_epoch_timings()
        timings["time/training and exploration (s)"] = self.total_train_expl_time
        logger.record_dict(timings)

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        pretrain_policy=None,
        num_pretrain_steps=0,
        use_pretrain_policy_for_initial_data=True,
        eval_buffer=None,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        if use_pretrain_policy_for_initial_data:
            self.pretrain_policy = pretrain_policy
        else:
            self.pretrain_policy = None
        self.num_pretrain_steps = num_pretrain_steps
        self.total_train_expl_time = 0
        self.eval_buffer = eval_buffer

    def _train(self):
        st = time.time()
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                runtime_policy=self.pretrain_policy,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        self.total_train_expl_time += time.time() - st
        self.trainer.buffer = self.replay_buffer  # TODO: make a cleaner of doing this
        self.training_mode(True)
        for _ in range(self.num_pretrain_steps):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            self.trainer.train(train_data)
        self.training_mode(False)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
            )
            gt.stamp("evaluation sampling")
            st = time.time()
            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                )
                gt.stamp("exploration sampling", unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)
                for train_step in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            if self.eval_buffer:
                eval_data = self.eval_buffer.random_batch(self.batch_size)
                self.trainer.evaluate(eval_data, buffer_data=False)
                eval_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.evaluate(eval_data, buffer_data=True)
            self.total_train_expl_time += time.time() - st

            self._end_epoch(epoch)

    def save(self):
        path = logger.get_snapshot_dir()
        trainer = self.trainer
        expl_data_collector = self.expl_data_collector
        eval_data_collector = self.eval_data_collector
        replay_buffer = self.replay_buffer
        expl_env = self.expl_env
        eval_env = self.eval_env
        pretrain_policy = self.pretrain_policy

        delattr(self, "trainer")
        delattr(self, "expl_data_collector")
        delattr(self, "eval_data_collector")
        delattr(self, "replay_buffer")
        delattr(self, "expl_env")
        delattr(self, "eval_env")
        delattr(self, "pretrain_policy")

        pickle.dump(self, open(os.path.join(path, "algorithm.pkl"), "wb"))

        trainer.save(path, "trainer.pkl")
        expl_data_collector.save(path, "expl_data_collector.pkl")
        eval_data_collector.save(path, "eval_data_collector.pkl")
        replay_buffer.save(path, "replay_buffer.pkl")
        expl_env.save(path, "expl_env.pkl")
        eval_env.save(path, "eval_env.pkl")
        pretrain_policy.save(path, "pretrain_policy.pkl")

        self.trainer = trainer
        self.expl_data_collector = expl_data_collector
        self.eval_data_collector = eval_data_collector
        self.replay_buffer = replay_buffer
        self.expl_env = expl_env
        self.eval_env = eval_env
        self.pretrain_policy = pretrain_policy

    def load(self):
        path = logger.get_snapshot_dir()
        algorithm = pickle.load(open(os.path.join(path, "algorithm.pkl"), "rb"))
        algorithm.trainer = self.trainer
        algorithm.expl_data_collector = self.expl_data_collector
        algorithm.eval_data_collector = self.eval_data_collector
        algorithm.replay_buffer = self.replay_buffer
        algorithm.expl_env = self.expl_env
        algorithm.eval_env = self.eval_env
        algorithm.pretrain_policy = self.pretrain_policy

        algorithm.trainer.load(path, "trainer.pkl")
        algorithm.expl_data_collector.load(path, "expl_data_collector.pkl")
        algorithm.eval_data_collector.load(path, "eval_data_collector.pkl")
        algorithm.replay_buffer.load(path, "replay_buffer.pkl")
        algorithm.expl_env.load(path, "expl_env.pkl")
        algorithm.eval_env.load(path, "eval_env.pkl")
        algorithm.pretrain_policy.load(path, "pretrain_policy.pkl")
        return algorithm

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)
        if (epoch + 1) % 100 == 0:
            # TODO: update this hardcoded quantity, just something large so you don't save every epoch
            self.save()


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class MultiManagerBatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        manager,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        pretrain_policies=None,
        num_pretrain_steps=0,
        use_pretrain_policy_for_initial_data=True,
        primitive_model_pretrain_trainer=None,
        rollouts=None,
        primitive_model_buffer=None,
        primitive_model_batch_size=0,
        primitive_model_trainer=None,
        primitive_model_num_pretrain_steps=0,
        primitive_model_num_trains_per_train_loop=1,
        primitive_model_path=None,
        collect_data_using_primitive_model=False,
        train_primitive_model=False,
        discount=1,
        primitive_learning_algorithm="gcsl",
    ):
        self._start_epoch = 0
        self.manager = manager
        self.post_epoch_funcs = []
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        if use_pretrain_policy_for_initial_data:
            self.pretrain_policies = pretrain_policies
        else:
            self.pretrain_policies = None
        self.num_pretrain_steps = num_pretrain_steps
        self.total_train_expl_time = 0
        self.primitive_model_pretrain_trainer = primitive_model_pretrain_trainer
        self.rollouts = rollouts
        self.primitive_model_buffer = primitive_model_buffer
        self.primitive_model_batch_size = primitive_model_batch_size
        self.primitive_model_num_pretrain_steps = primitive_model_num_pretrain_steps
        self.primitive_model_num_trains_per_train_loop = (
            primitive_model_num_trains_per_train_loop
        )
        self.primitive_model_path = primitive_model_path
        self.collect_data_using_primitive_model = collect_data_using_primitive_model
        self.train_primitive_model = train_primitive_model
        self.discount = discount
        self.primitive_learning_algorithm = primitive_learning_algorithm
        self.primitive_model_trainer = primitive_model_trainer

    def _train(self):
        print("Initial Primitive Model Sync")
        torch.save(
            self.primitive_model_pretrain_trainer.policy.state_dict(),
            self.primitive_model_path,
        )
        self.manager.sync_primitive_model()
        st = time.time()
        print("Initial RAPS Data Collection")
        if self.min_num_steps_before_training > 0:
            for i in range(self.min_num_steps_before_training // 100):
                init_expl_paths = self.manager.collect_init_expl_paths(
                    self.min_num_steps_before_training // 100
                )
                for paths in init_expl_paths:
                    self.primitive_model_buffer.add_paths(paths)
        print("Manager Pretraining")
        self.manager.pretrain()
        print("Primitive Model Pretraining")
        self.training_mode(True)
        for train_step in range(self.primitive_model_num_pretrain_steps):
            train_data = self.primitive_model_buffer.random_batch(
                self.primitive_model_batch_size
            )
            self.primitive_model_pretrain_trainer.train(train_data)
        self.training_mode(False)
        self.total_train_expl_time += time.time() - st
        print("Primitive Model Sync")
        torch.save(
            self.primitive_model_pretrain_trainer.policy.state_dict(),
            self.primitive_model_path,
        )
        self.manager.sync_primitive_model()
        if self.collect_data_using_primitive_model:
            self.manager.set_use_primitive_model()
        self._end_epoch(-1)
        # clear primitive model buffer:
        # self.primitive_model_buffer._size = 0
        # self.primitive_model_buffer._top = 0
        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            print("Eval Data Collection")
            self.manager.collect_eval_paths()
            gt.stamp("evaluation sampling")
            st = time.time()
            for _ in range(self.num_train_loops_per_epoch):
                print("Expl Data Collection")
                expl_paths = self.manager.collect_expl_paths()
                gt.stamp("exploration sampling", unique=False)
                if self.primitive_learning_algorithm == "gcsl":
                    for paths in expl_paths:
                        self.primitive_model_buffer.add_paths(paths)
                elif self.primitive_learning_algorithm == "ppo":
                    setup_ppo_buffer(
                        expl_paths, self.discount, self.policy, self.rollouts
                    )
                gt.stamp("storing", unique=False)
                print("Manager Training")
                self.manager.train()
                gt.stamp("manager training", unique=False)
                print("Primitive Model Training")
                if self.train_primitive_model:
                    if self.primitive_learning_algorithm == "gcsl":
                        self.training_mode(True)
                        for train_step in range(
                            self.primitive_model_num_trains_per_train_loop
                        ):
                            train_data = self.primitive_model_buffer.random_batch(
                                self.primitive_model_batch_size
                            )
                            self.primitive_model_trainer.train(train_data)
                        self.training_mode(False)
                    elif self.primitive_learning_algorithm == "ppo":
                        self.rollouts.to(ptu.device)

                        with torch.no_grad():
                            next_value = (
                                self.primitive_model_pretrain_trainer.policy.get_value(
                                    self.rollouts.obs[-1],
                                    self.rollouts.recurrent_hidden_states[-1],
                                    self.rollouts.masks[-1],
                                ).detach()
                            )
                        self.rollouts.compute_returns(
                            next_value, **self.rollouts.rollout_kwargs
                        )
                        self.primitive_model_trainer.update(self.rollouts)
                        self.rollouts.after_update()
                        self.rollouts.to(torch.device("cpu"))
                self.total_train_expl_time += time.time() - st
                gt.stamp("primitive model training", unique=False)
                torch.save(
                    self.primitive_model_pretrain_trainer.policy.state_dict(),
                    self.primitive_model_path,
                )
                self.manager.sync_primitive_model()
                gt.stamp("primitive model syncing", unique=False)
            self.total_train_expl_time += time.time() - st
            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        if epoch > -1:
            snapshot = self._get_snapshot()
            logger.save_itr_params(epoch, snapshot)
            gt.stamp("saving")
            self._log_stats(epoch)

            # self.manager._end_epoch(epoch)
            self.manager.save(logger.get_snapshot_dir())

            if self.primitive_learning_algorithm == "gcsl":
                self.primitive_model_buffer.end_epoch(epoch)
                self.primitive_model_trainer.end_epoch(epoch)

            for post_epoch_func in self.post_epoch_funcs:
                post_epoch_func(self, epoch)
        else:
            for post_epoch_func in self.post_epoch_funcs:
                post_epoch_func(self, epoch)
            self.primitive_model_pretrain_trainer.end_epoch(epoch)

    def load(self, path):
        path = os.path.join(
            "/home/mdalal/research/skill_learn/rlkit/", path
        )  # TODO: fix hardcoded path
        self.manager.load(path)
        # self.manager.sync_primitive_model_from_path(
        #     os.path.join(path, "primitive_model.ptc")
        # )

    def _get_snapshot(self):
        snapshot = {}
        return snapshot

    def _log_stats(self, epoch):
        logger.log(f"Epoch {epoch} finished", with_timestamp=True)

        # self.manager._log_stats()

        logger.record_dict(
            self.primitive_model_pretrain_trainer.get_diagnostics(),
            prefix=f"primitive_model_pretrain_trainer/",
        )

        logger.record_dict(
            self.primitive_model_trainer.get_diagnostics(),
            prefix=f"primitive_model_trainer/",
        )
        if self.train_primitive_model:
            if self.primitive_learning_algorithm == "gcsl":
                logger.record_dict(
                    self.primitive_model_buffer.get_diagnostics(),
                    prefix=f"primitive_model_buffer/",
                )
            elif self.primitive_learning_algorithm == "ppo":
                logger.record_dict(
                    self.rollouts.get_diagnostics(),
                    prefix=f"primitive_model_buffer/",
                )

        """
        Misc
        """
        gt.stamp("logging")
        timings = _get_epoch_timings()
        timings["time/training and exploration (s)"] = self.total_train_expl_time
        logger.record_dict(timings)
        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)


class TorchMultiManagerBatchRLAlgorithm(MultiManagerBatchRLAlgorithm):
    def to(self, device):
        self.primitive_model_pretrain_trainer.to(device)

    def training_mode(self, mode):
        for net in self.primitive_model_pretrain_trainer.networks:
            net.train(mode)


def setup_ppo_buffer(expl_paths, discount, policy, rollouts):
    obs = []
    actions = []
    actor_log_probs = []
    values = []
    rewards = []
    masks = []
    for paths in expl_paths:
        for path in paths:
            high_level_reward = np.expand_dims(path["rewards"], -1)
            low_level_reward = path["low_level_rewards"]
            reward_input = high_level_reward[1:-1, :, :]
            value = np.zeros_like(reward_input)
            discount = np.ones_like(reward_input) * discount
            bootstrap = reward_input[-1, :, :]
            lambda_ = 1
            returns = lambda_return_np(
                reward_input, value, discount, bootstrap, lambda_
            )
            returns = np.concatenate((returns, np.expand_dims(bootstrap, 0)), axis=0)
            returns = returns.reshape(-1, 1, 1)[:, 0]
            low_level_reward[:, -1] += returns
            for i in range(0, path["low_level_observations"].shape[1]):
                if i > 0:
                    inputs = ptu.from_numpy(
                        np.concatenate(
                            (
                                path["low_level_observations"][:, i - 1],
                                path["high_level_actions"][:, i - 1],
                            ),
                            axis=-1,
                        )
                    )
                    (value, action_log_probs, _, _,) = policy.evaluate_actions(
                        inputs,
                        ptu.zeros_like(inputs),
                        ptu.zeros_like(inputs),
                        ptu.from_numpy(path["low_level_actions"][:, i - 1]),
                    )
                    value = value.cpu().detach()
                    action_log_probs = action_log_probs.cpu().detach()
                if len(obs) <= i:
                    if i > 0:
                        obs.append(
                            torch.from_numpy(
                                np.concatenate(
                                    (
                                        path["low_level_observations"][:, i],
                                        path["high_level_actions"][:, i - 1],
                                    ),
                                    axis=-1,
                                )
                            )
                        )
                    else:
                        obs.append(
                            torch.from_numpy(
                                np.concatenate(
                                    (
                                        path["low_level_observations"][:, i],
                                        path["high_level_actions"][:, i],
                                    ),
                                    axis=-1,
                                )
                            )
                        )
                    if i > 0:
                        actions.append(
                            torch.from_numpy(path["low_level_actions"][:, i - 1])
                        )
                        rewards.append(torch.from_numpy(low_level_reward[:, i - 1]))
                        masks.append(
                            torch.from_numpy(path["low_level_terminals"][:, i - 1])
                        )
                        values.append(value)
                        actor_log_probs.append(action_log_probs)
                else:
                    if i > 0:
                        obs[i] = torch.cat(
                            (
                                obs[i],
                                torch.from_numpy(
                                    np.concatenate(
                                        (
                                            path["low_level_observations"][:, i],
                                            path["high_level_actions"][:, i - 1],
                                        ),
                                        axis=-1,
                                    ),
                                ),
                            ),
                            dim=0,
                        )
                    else:
                        obs[i] = torch.cat(
                            (
                                obs[i],
                                torch.from_numpy(
                                    np.concatenate(
                                        (
                                            path["low_level_observations"][:, i],
                                            path["high_level_actions"][:, i],
                                        ),
                                        axis=-1,
                                    ),
                                ),
                            ),
                            dim=0,
                        )
                    if i > 0:
                        actions[i - 1] = torch.cat(
                            (
                                actions[i - 1],
                                torch.from_numpy(path["low_level_actions"][:, i - 1]),
                            ),
                            axis=0,
                        )
                        rewards[i - 1] = torch.cat(
                            (
                                rewards[i - 1],
                                torch.from_numpy(low_level_reward[:, i - 1]),
                            ),
                            dim=0,
                        )
                        masks[i - 1] = torch.cat(
                            (
                                masks[i - 1],
                                torch.from_numpy(path["low_level_terminals"][:, i - 1]),
                            ),
                            dim=0,
                        )
                        values[i - 1] = torch.cat((values[i - 1], value), dim=0)
                        actor_log_probs[i - 1] = torch.cat(
                            (
                                actor_log_probs[i - 1],
                                action_log_probs,
                            ),
                            dim=0,
                        )
    rollouts.obs[0] = obs[0]
    for step, (o, a, al, v, r, m) in enumerate(
        zip(obs[1:], actions, actor_log_probs, values, rewards, masks)
    ):
        rollouts.insert(
            o,
            torch.zeros_like(o),
            a,
            al,
            v,
            r,
            m,
            m,
            torch.ones_like(r[:, 0]) * step,
        )
