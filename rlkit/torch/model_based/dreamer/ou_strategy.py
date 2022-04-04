import numpy as np
import numpy.random as nr
import torch
from torch import jit
import rlkit.torch.pytorch_util as ptu
    
@jit.script
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
        dim:int,
        low:torch.Tensor,
        high:torch.Tensor,
        mu:int=0,
        theta:float=0.15,
        max_sigma:float=0.3,
        min_sigma:float=0,
        decay_period:float=100000,
    ):
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = torch.tensor(mu)
        self.theta = torch.tensor(theta)
        self.sigma = torch.tensor(max_sigma)
        self._max_sigma = max_sigma
        if min_sigma == 0:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self.dim = torch.tensor(dim)
        self.low = torch.tensor(low).to(ptu.device)
        self.high = torch.tensor(high).to(ptu.device)
        
        self.state = torch.ones(self.dim) * self.mu
    
    
    def reset(self):
        self.state = torch.ones(self.dim) * self.mu

    
    def evolve_state(self):
        with torch.no_grad():
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * torch.randn(len(x))
            self.state = x + dx
            return self.state

    
    def get_action_from_raw_action(self, action:torch.Tensor, t:int=0):
        with torch.no_grad():
            ou_state = self.evolve_state()
            self.sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * torch.tensor(min(
                1.0, t * 1.0 / self._decay_period
            ))

            ou_state = ou_state.to(ptu.device)
            new_action = action + ou_state
            new_action = torch.where(new_action < self.low, self.low, new_action)
            new_action = torch.where(new_action > self.high, self.high, new_action)

            return new_action
