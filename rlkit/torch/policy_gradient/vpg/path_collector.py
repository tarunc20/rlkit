from collections import OrderedDict, deque

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.torch.policy_gradient.vpg import vec_rollout

class VectorizedMdpPathCollector(PathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        rollout_fn=vec_rollout,
        save_env_in_snapshot=False,
        env_params=None,
        env_class=None,
        rollout_function_kwargs=None
    ):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._rollout_fn = rollout_fn


        self._num_steps_total = 0
        self._num_path_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot
        self.env_params = env_params
        self.env_class = env_class
        self.rollout_function_kwargs = rollout_function_kwargs

    def collect_new_paths(
        self,
        max_path_length,
        num_steps,
        runtime_policy=None
    ):
        paths = []
        num_steps_collected = 0

        while num_steps_collected < num_steps:
            
            if not runtime_policy:
                runtime_policy = self._policy
            
            path = self._rollout_fn(
                self._env, 
                runtime_policy,
                max_path_length=max_path_length,
                rollout_function_kwargs=self.rollout_function_kwargs
            )

            path_len = len(path["actions"])
            num_steps_collected += path_len * self._env.n_envs
            paths.append(path)

            self._num_paths_total += len(paths) * self._env.n_envs
            self._num_steps_total += num_steps_collected
            log_paths = [{} for _ in range(len(paths) * self.env.n_envs)]

            count = 0
            for path in paths:
                for env_idx in range(self._env.n_envs):
                    for key in [
                        "actions",
                        "terminals",
                        "rewards"
                    ]:
                        # Skip the first action as it is null
                        log_paths[count][key] = path[key][1:, env_idx]

                    # TODO: Figure out what each dimension means
                    log_paths[count]["agent_infos"] = [{}] * path["rewards"][
                        1:, env_idx
                    ].shape[0]

                    env_info_key = "env_infos"
                    log_paths[count][env_info_key] = [{}] * path["rewards"][
                        1:, env_idx
                    ].shape[0]

                    for key, value in path[env_info_key].items():
                        for value_idx in range(value[env_idx].shape[0]):
                            log_paths[count][env_info_key][value_idx][key] = value[env_idx][
                                value_idx
                            ]
                    
                    count += 1

                self._epoch_paths.extend(log_paths)

                return paths

def get_epoch_paths(self):
    return self._epoch_paths

def end_epoch(self, epoch):
    self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

def get_diagnostics(self):
    path_lens = [len(path["actions"]) for path in self._epoch_paths]

    stats = OrderedDict(
        [
            ("num steps total", self._num_steps_total),
            ("num paths total", self._num_paths_total)
        ]
    )

    stats.update(
        create_stats_ordered_dict(
            "path length",
            path_lens, 
            always_show_all_stats=True
        )
    )

    return stats