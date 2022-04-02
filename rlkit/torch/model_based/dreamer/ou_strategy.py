import numpy as np
import numpy.random as nr
import torch
from torch import jit

class OUStrategy():
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    Based on the rllab implementation.
    """

    def __init__(
        self,
        dim,
        low,
        high,
        mu=0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=None,
        decay_period=100000,
    ):
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = torch.Tensor(mu)
        self.theta = theta
        self.sigma = max_sigma
        self._max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self.dim = dim
        self.low = low
        self.high = high
        self.state = torch.ones(self.dim) * self.mu
        self.reset()

    def reset(self):
        self.state = torch.ones(self.dim) * self.mu

    @jit.script_method
    def evolve_state(self):
        with torch.no_grad():
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * torch.randn(len(x))
            self.state = x + dx
            return self.state

    @jit.script_method
    def get_action_from_raw_action(self, action, t=0):
        with torch.no_grad():
            ou_state = self.evolve_state()
            self.sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
                1.0, t * 1.0 / self._decay_period
            )
            return torch.clip(action + ou_state, self.low, self.high)
