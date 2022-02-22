import multiprocessing as mp
import os

from stable_baselines3.common.vec_env import CloudpickleWrapper

import rlkit.torch.pytorch_util as ptu
from rlkit.core import eval_util, logger


class Manager:
    def __init__(
        self,
        expl_env,
        eval_env,
        expl_env_path_collector,
        eval_env_path_collector,
        trainer,
        replay_buffer,
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
        manager_idx=0,
    ):
        self.expl_env = expl_env
        self.eval_env = eval_env
        self.expl_env_path_collector = expl_env_path_collector
        self.eval_env_path_collector = eval_env_path_collector
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training
        self.pretrain_policy = pretrain_policy
        self.num_pretrain_steps = num_pretrain_steps
        self.manager_idx = manager_idx

    def training_mode(self, mode):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        for net in self.trainer.networks:
            net.train(mode)

    def collect_init_expl_paths(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        init_expl_paths = self.expl_env_path_collector.collect_new_paths(
            self.max_path_length,
            self.min_num_steps_before_training,
            runtime_policy=self.pretrain_policy,
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
                primitive_model_path[primitive_model_key] = path[primitive_model_key]
            manager_paths.append(manager_path)
            primitive_model_paths.append(primitive_model_path)
        self.replay_buffer.add_paths(manager_paths)
        self.expl_env_path_collector.end_epoch(-1)
        return primitive_model_paths

    def pretrain(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        self.training_mode(True)
        for _ in range(self.num_pretrain_steps):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            self.trainer.train(train_data)
        self.training_mode(False)

    def collect_eval_paths(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        self.eval_env_path_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
        )

    def train(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        self.training_mode(True)
        for train_step in range(self.num_trains_per_train_loop):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            self.trainer.train(train_data)
        self.training_mode(False)

    def collect_expl_paths(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        new_expl_paths = self.expl_env_path_collector.collect_new_paths(
            self.max_path_length,
            self.num_expl_steps_per_train_loop,
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
        for path in new_expl_paths:
            manager_path = {}
            primitive_model_path = {}
            for manager_key in manager_keys:
                manager_path[manager_key] = path[manager_key]
            for primitive_model_key in primitive_model_keys:
                primitive_model_path[primitive_model_key] = path[primitive_model_key]
            manager_paths.append(manager_path)
            primitive_model_paths.append(primitive_model_path)
        self.replay_buffer.add_paths(manager_paths)
        return primitive_model_paths

    def _end_epoch(self, epoch):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        self.expl_env_path_collector.end_epoch(epoch)
        self.eval_env_path_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

    def _log_stats(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        expl_paths = self.expl_env_path_collector.get_epoch_paths()
        eval_paths = self.eval_env_path_collector.get_epoch_paths()
        return (
            self.replay_buffer.get_diagnostics(),
            self.trainer.get_diagnostics(),
            self.expl_env_path_collector.get_diagnostics(),
            eval_util.get_generic_path_information(expl_paths),
            self.eval_env_path_collector.get_diagnostics(),
            eval_util.get_generic_path_information(eval_paths),
        )

    def sync_primitive_model(self):
        ptu.set_gpu_mode(True, gpu_id=self.manager_idx)
        self.expl_env.sync_primitive_model()
        self.eval_env.sync_primitive_model()

    def get_obs_and_action_dims(self):
        return (
            self.eval_env.observation_space.low.size,
            self.eval_env.action_space.low.size,
        )


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    manager_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    manager = manager_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "collect_init_expl_paths":
                remote.send(manager.collect_init_expl_paths())
            elif cmd == "pretrain":
                remote.send(manager.pretrain())
            elif cmd == "collect_eval_paths":
                remote.send(manager.collect_eval_paths())
            elif cmd == "train":
                remote.send(manager.train())
            elif cmd == "close":
                remote.send(remote.close())
                break
            elif cmd == "collect_expl_paths":
                remote.send(manager.collect_expl_paths())
            elif cmd == "_end_epoch":
                remote.send(manager._end_epoch(int(data)))
            elif cmd == "_log_stats":
                remote.send(manager._log_stats())
            elif cmd == "set_process_gpu_device_id":
                os.environ["EGL_DEVICE_ID"] = str(data)
                ptu.set_gpu_mode(True, gpu_id=int(data))
                manager.device_id = int(data)
            elif cmd == "sync_primitive_model":
                remote.send(manager.sync_primitive_model())
            elif cmd == "get_obs_and_action_dims":
                remote.send(manager.get_obs_and_action_dims())
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class VecManager:
    def __init__(
        self,
        manager_fns,
        env_names,
        start_method=None,
        reload_state_args=None,
        device_id=0,
    ):
        self.device_id = 0
        self.waiting = False
        self.closed = False
        self.reload_state_args = reload_state_args
        n_managers = len(manager_fns)
        self.n_managers = n_managers
        self.env_names = env_names

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context("spawn")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_managers)])
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, manager_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=False
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
        for remote in self.remotes:
            remote.send(("set_process_gpu_device_id", device_id))

    def set_primitive_model_buffer(self, primitive_model_buffer):
        self.primitive_model_buffer = primitive_model_buffer

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_obs_and_action_dims(self):
        self.remotes[0].send(("get_obs_and_action_dims", None))
        self.waiting = True
        out = self.remotes[0].recv()
        self.waiting = False
        return out

    def sync_primitive_model(self):
        for remote in self.remotes:
            remote.send(("sync_primitive_model", None))
            self.waiting = True
        for remote in self.remotes:
            remote.recv()
        self.waiting = False

    def collect_init_expl_paths(self):
        for remote in self.remotes:
            remote.send(("collect_init_expl_paths", None))
            self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        for result in results:
            self.primitive_model_buffer.add_paths(result)

    def pretrain(self):
        for remote in self.remotes:
            remote.send(("pretrain", None))
            self.waiting = True
        for remote in self.remotes:
            remote.recv()
        self.waiting = False

    def collect_eval_paths(self):
        for remote in self.remotes:
            remote.send(("collect_eval_paths", None))
            self.waiting = True
        for remote in self.remotes:
            remote.recv()
        self.waiting = False

    def train(self):
        for remote in self.remotes:
            remote.send(("train", None))
            self.waiting = True
        for remote in self.remotes:
            remote.recv()
        self.waiting = False

    def collect_expl_paths(self):
        for remote in self.remotes:
            remote.send(("collect_expl_paths", None))
            self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        for result in results:
            self.primitive_model_buffer.add_paths(result)

    def _end_epoch(self, epoch):
        for remote in self.remotes:
            remote.send(("_end_epoch", epoch))
            self.waiting = True
        for remote in self.remotes:
            remote.recv()
        self.waiting = False

    def _log_stats(self):
        for manager_idx, remote in enumerate(self.remotes):
            remote.send(("_log_stats", None))
            self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        for manager_idx, result in enumerate(results):
            (
                buffer_diagnostics,
                trainer_diagnostics,
                expl_env_path_collector_diagnostics,
                expl_paths,
                eval_env_path_collector_diagnostics,
                eval_paths,
            ) = result
            """
            Replay Buffer
            """
            logger.record_dict(
                buffer_diagnostics,
                prefix=f"{self.env_names[manager_idx]}/replay_buffer/",
            )

            """
            Trainer
            """
            logger.record_dict(
                trainer_diagnostics,
                prefix=f"{self.env_names[manager_idx]}/trainer/",
            )

            """
            Exploration
            """
            logger.record_dict(
                expl_env_path_collector_diagnostics,
                prefix=f"{self.env_names[manager_idx]}/exploration/",
            )

            logger.record_dict(
                expl_paths,
                prefix=f"{self.env_names[manager_idx]}/exploration/",
            )

            """
            Evaluation
            """
            logger.record_dict(
                eval_env_path_collector_diagnostics,
                prefix=f"{self.env_names[manager_idx]}/evaluation/",
            )

            logger.record_dict(
                eval_paths,
                prefix=f"{self.env_names[manager_idx]}/evaluation/",
            )
