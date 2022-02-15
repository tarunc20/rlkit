import abc
import os
import pickle
import time
from collections import OrderedDict

import gtimer as gt

import rlkit.torch.pytorch_util as ptu
from rlkit.core import eval_util, logger
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector, PathCollector


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
        trainers,
        exploration_envs,
        evaluation_envs,
        exploration_data_collectors,
        evaluation_data_collectors,
        replay_buffers,
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
        eval_buffers=None,
        env_names=None,
        primitive_model_pretrain_trainer=None,
        primitive_model_trainer=None,
        primitive_model_buffer=None,
    ):
        self.trainers = trainers
        self.expl_envs = exploration_envs
        self.eval_envs = evaluation_envs
        self.expl_data_collectors = exploration_data_collectors
        self.eval_data_collectors = evaluation_data_collectors
        self.replay_buffers = replay_buffers
        self._start_epoch = 0

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
        self.eval_buffers = eval_buffers
        self.num_managers = len(self.trainers)
        self.env_names = env_names
        self.primitive_model_pretrain_trainer = primitive_model_pretrain_trainer
        self.primitive_model_trainer = primitive_model_trainer
        self.primitive_model_buffer = primitive_model_buffer

    def _train(self):
        st = time.time()
        if self.min_num_steps_before_training > 0:
            for manager_idx in range(self.num_managers):
                ptu.set_gpu_mode(True, gpu_id=manager_idx)
                os.environ["EGL_DEVICE_ID"] = str(manager_idx)
                init_expl_paths = self.expl_data_collectors[
                    manager_idx
                ].collect_new_paths(
                    self.max_path_length,
                    self.min_num_steps_before_training,
                    runtime_policy=self.pretrain_policies[manager_idx],
                )
                manager_keys = ["observations", "actions", "rewards", "terminals"]
                primitive_model_keys = [
                    "low_level_observations",
                    "high_level_actions",
                    "low_level_actions",
                    "low_level_rewards",
                    "low_level_terminals",
                ]
                manager_paths = []
                primitive_model_paths = []
                for path in init_expl_paths:
                    manager_path = {}
                    primitive_model_path = {}
                    for manager_key in manager_keys:
                        manager_path[manager_key] = path[manager_key]
                    for primitive_model_key in primitive_model_keys:
                        primitive_model_path[primitive_model_key] = path[
                            primitive_model_key
                        ]
                    manager_paths.append(manager_path)
                    primitive_model_paths.append(primitive_model_path)
                self.replay_buffers[manager_idx].add_paths(manager_paths)
                self.primitive_model_buffer.add_paths(primitive_model_paths)
                self.expl_data_collectors[manager_idx].end_epoch(-1)
        self.total_train_expl_time += time.time() - st
        for manager_idx in range(self.num_managers):
            ptu.set_gpu_mode(True, gpu_id=manager_idx)
            os.environ["EGL_DEVICE_ID"] = str(manager_idx)
            self.trainers[manager_idx].buffer = self.replay_buffers[
                manager_idx
            ]  # TODO: make a cleaner of doing this
            self.training_mode(True)
            for _ in range(self.num_pretrain_steps):
                train_data = self.replay_buffers[manager_idx].random_batch(
                    self.batch_size
                )
                self.trainers[manager_idx].train(train_data)
            self.training_mode(False)
        ptu.set_gpu_mode(True, gpu_id=3)
        for _ in range(self.num_pretrain_steps * self.num_managers):
            train_data = self.primitive_model_buffer.random_batch(self.batch_size)
            self.primitive_model_pretrain_trainer.train(train_data)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            for manager_idx in range(self.num_managers):
                ptu.set_gpu_mode(True, gpu_id=manager_idx)
                os.environ["EGL_DEVICE_ID"] = str(manager_idx)
                self.eval_data_collectors[manager_idx].collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                )
                gt.stamp("evaluation sampling", unique=False)
                st = time.time()
                for _ in range(self.num_train_loops_per_epoch):
                    new_expl_paths = self.expl_data_collectors[
                        manager_idx
                    ].collect_new_paths(
                        self.max_path_length,
                        self.num_expl_steps_per_train_loop,
                    )
                    gt.stamp("exploration sampling", unique=False)

                    manager_keys = ["observations", "actions", "rewards", "terminals"]
                    primitive_model_keys = [
                        "low_level_observations",
                        "high_level_actions",
                        "low_level_actions",
                        "low_level_rewards",
                        "low_level_terminals",
                    ]
                    manager_paths = []
                    primitive_model_paths = []
                    for path in init_expl_paths:
                        manager_path = {}
                        primitive_model_path = {}
                        for manager_key in manager_keys:
                            manager_path[manager_key] = path[manager_key]
                        for primitive_model_key in primitive_model_keys:
                            primitive_model_path[primitive_model_key] = path[
                                primitive_model_key
                            ]
                        manager_paths.append(manager_path)
                        primitive_model_paths.append(primitive_model_path)
                    self.replay_buffers[manager_idx].add_paths(manager_paths)
                    self.primitive_model_buffer.add_paths(primitive_model_paths)
                    gt.stamp("data storing", unique=False)

                    self.training_mode(True)
                    for train_step in range(self.num_trains_per_train_loop):
                        train_data = self.replay_buffers[manager_idx].random_batch(
                            self.batch_size
                        )
                        self.trainers[manager_idx].train(train_data)
                    gt.stamp("training", unique=False)
                    self.training_mode(False)

                if self.eval_buffers[0] is not None:
                    eval_data = self.eval_buffers[manager_idx].random_batch(
                        self.batch_size
                    )
                    self.trainers[manager_idx].evaluate(eval_data, buffer_data=False)
                    eval_data = self.replay_buffers[manager_idx].random_batch(
                        self.batch_size
                    )
                    self.trainers[manager_idx].evaluate(eval_data, buffer_data=True)
                self.total_train_expl_time += time.time() - st

            st = time.time()
            ptu.set_gpu_mode(True, gpu_id=3)
            for train_step in range(
                self.num_trains_per_train_loop
                * self.num_train_loops_per_epoch
                * self.num_managers
            ):
                train_data = self.primitive_model_buffer.random_batch(self.batch_size)
                self.primitive_model_trainer.train(train_data)
            self.total_train_expl_time += time.time() - st

            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp("saving")
        self._log_stats(epoch)

        for manager_idx in range(self.num_managers):
            self.expl_data_collectors[manager_idx].end_epoch(epoch)
            self.eval_data_collectors[manager_idx].end_epoch(epoch)
            self.replay_buffers[manager_idx].end_epoch(epoch)
            self.trainers[manager_idx].end_epoch(epoch)

        # for post_epoch_func in self.post_epoch_funcs:
        #     post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        return snapshot

    def _log_stats(self, epoch):
        logger.log(f"Epoch {epoch} finished", with_timestamp=True)

        for manager_idx in range(self.num_managers):
            """
            Replay Buffer
            """
            logger.record_dict(
                self.replay_buffers[manager_idx].get_diagnostics(),
                prefix=f"{self.env_names[manager_idx]}/replay_buffer/",
            )

            """
            Trainer
            """
            logger.record_dict(
                self.trainers[manager_idx].get_diagnostics(),
                prefix=f"{self.env_names[manager_idx]}/trainer/",
            )

            """
            Exploration
            """
            logger.record_dict(
                self.expl_data_collectors[manager_idx].get_diagnostics(),
                prefix=f"{self.env_names[manager_idx]}/exploration/",
            )

            expl_paths = self.expl_data_collectors[manager_idx].get_epoch_paths()
            if len(expl_paths) > 0:
                logger.record_dict(
                    eval_util.get_generic_path_information(expl_paths),
                    prefix=f"{self.env_names[manager_idx]}/exploration/",
                )

            """
            Evaluation
            """
            logger.record_dict(
                self.eval_data_collectors[manager_idx].get_diagnostics(),
                prefix=f"{self.env_names[manager_idx]}/evaluation/",
            )
            eval_paths = self.eval_data_collectors[manager_idx].get_epoch_paths()

            logger.record_dict(
                eval_util.get_generic_path_information(eval_paths),
                prefix=f"{self.env_names[manager_idx]}/evaluation/",
            )

        logger.record_dict(
            self.primitive_model_trainer.get_diagnostics(),
            prefix=f"primitive_model_trainer/trainer/",
        )

        """
        Misc
        """
        gt.stamp("logging")
        timings = _get_epoch_timings()
        logger.record_dict(timings)
        timings["time/training and exploration (s)"] = self.total_train_expl_time
        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)


class TorchMultiManagerBatchRLAlgorithm(MultiManagerBatchRLAlgorithm):
    def to(self, device):
        for trainer in self.trainers:
            for net in trainer.networks:
                net.to(device)

    def training_mode(self, mode):
        for trainer in self.trainers:
            for net in trainer.networks:
                net.train(mode)
