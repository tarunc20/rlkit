import abc
from enum import unique
import time
from typing import OrderedDict
import gtimer as gt
from pyparsing import with_attribute

from rlkit.core import eval_util, logger
from rlkit.samplers.data_collector.base import DataCollector, PathCollector
from rlkit.data_management.replay_buffer import ReplayBuffer

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
        env,
        data_collector: DataCollector,
        replay_buffer: ReplayBuffer
    ):
        self.trainer = trainer
        self.env = env
        self.data_collector = data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch= 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()
    
    def _train(self):
        '''
        Trian model
        '''
        raise NotImplementedError('_train should be be implemented by the inherited class')

    def _get_snapshot(self):
        snapshot = {}
        for key, value in self.trainer.get_snapshot().items():
            snapshot["trainer/" + key] = value
        for key, value in self.expl_data_collector.get_snapshot().items():
            snapshot["exploration/" + key] = value
        for key, value in self.eval_data_collector.get_snapshot().items():
            snapshot["evaluation/" + key] = value
        for key, value in self.replay_buffer.get_snapshot().items():
            snapshot["replay_buffer/"+ key] = value
        return snapshot

    def _log_stats(self, epoch):
        logger.log(f"Epoch {epoch} finished", with_timestamp=True)

        # Replay Buffer
        logger.record_dict(
            self.replay_buffer.get_diagnostics(), prefix="replay_buffer/"
        )

        # Trainer
        logger.record_dict(self.trainer.get_diagnostics(), prefix="trainder/")

        # Exploration
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(), prefix="exploration/"
        )

        expl_paths = self.expl_data_collector.get_epoch_paths()

        if len(expl_paths) > 0:
            if hasattr(self.expl_env, "get_diagnostics"):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix="exploration/"
                )

            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/"
            )
        
        # Evaluation
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix="evaluation/"
        )

        eval_paths = self.eval_data_collector.get_epoch_paths()
        
        if hasattr(self.eval_env, "get_diagnostics"):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix="evaluation/",
            )
        
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/"
        )

        # Misc
        gt.stamp("logging")
        timings = _get_epoch_timings()
        timings["time/training and exploration (s)"] = self.total_train_expl_time
        logger.record_dict(timings)

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

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
    
    @abc.abstractmethod
    def training_mode(self, mode):
        '''
        Set training mode to `mode`
        :param mode: If True, training will happen (e.g.
        set dropout probabilities to not all ones)
        '''
        pass

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        env,
        data_collector: PathCollector,
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
        eval_buffer=None
    ):
        super().__init__(
            trainer,
            env,
            data_collector,
            replay_buffer
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
        #TODO: Why use time and not gtimer
        st = time.time()

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                runtime_policy=self.pretrain_policy
            )

            self.replay_buffer.add_paths(
                init_expl_paths
            )

            self.data_collector.end_epoch(-1)

        self.total_train_time += time.time() - st
        self.trainer.buffer = self.replay_buffer

        #TODO: Do I still need this?
        self.training_mode(True)
        for _ in range(self.num_pretrain_steps):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            self.trainer.train(train_data)
        self.training_mode(False)        

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True
        ):
            st = time.time()

            for _ in range(self.num_train_loops_per_epoch):
                new_paths = self.data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop
                )

                gt.stamp('environment sampling', unique=False)

                self.replay_buffer.add_paths(new_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)
                for train_step in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                
                gt.stamp('training', unique=False)
                self.training_mode(False)
            
            self.total_train_expl_time += time.time() - st

            self._end_epoch(epoch)

class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
    
    def training_mode(self, mode):
        for net in self.trainer.networks:
            for net in self.trainer.networks:
                net.train(mode)



