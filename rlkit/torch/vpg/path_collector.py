
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
        self._num_low_level_steps_total = 0
        self._num_low_level_steps_total_true = 0
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