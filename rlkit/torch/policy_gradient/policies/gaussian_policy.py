from multiprocessing.sharedctypes import Value
from rlkit.torch.distributions import MultivariateDiagonalNormal
from rlkit.torch.sac.policies.gaussian_policy import LOG_SIG_MAX, LOG_SIG_MIN
import torch
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.networks import Mlp
from torch import nn
import numpy as np
import rlkit.torch.pytorch_util as ptu

LOG_SIG_MIN = 2
LOG_SIG_MAX = -20

class GaussianPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        std=None,
        init_w=1e-3,
        min_log_std=None,
        max_log_std=None,
        std_architecture="shared",
        **kwargs
    ):
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
        self.std = std
        self.log_std = None
        self.std_architecture = std_architecture

        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim

                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True)
                )
            
            else:
                raise ValueError(self.std_architecture)

        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <=LOG_SIG_MAX

    def forward(self, obs):
        hidden = obs
        
        for fc in self.fcs:
            hidden = self.hidden_activation(fc(hidden))

        preactivation = self.last_fc(hidden)
        mean = self.last_fc(preactivation)

        if self.std is None:
            if self.std_architecture == "shared":
                log_std = self.last_fc_log_std(hidden)
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)

        else:
            std = (
                torch.from_numpy(
                    np.array[
                        self.std
                    ]
                )
                .float()
                .to(ptu.device)
            )
        
        return MultivariateDiagonalNormal(mean, std)