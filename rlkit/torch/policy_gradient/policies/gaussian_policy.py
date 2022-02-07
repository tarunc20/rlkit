import torch
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.networks import Mlp

class GaussianPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        min_log_std=None,
        max_log_std=None,
        **kwargs
    ):
        #TODO Why tanh?
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
