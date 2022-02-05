from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.loss import LossFunction, LossStatistics

import torch.optim as optim
import gtimer as gt

class VPGTrainer(TorchTrainer, LossFunction):
    def __init__(
        self, 
        env,
        policy,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-2,
        optimizer_class=optim.Adam,
        plotter=None
    ):
        super().__init__()
        self.env = env
        self.policy = policy

        self.plotter = plotter
        
        self.policy_optimizer = optimizer_class(
            self.policy_parameters(),
            lr=policy_lr
        )

        self.discount = discount
        self.reward_scale = reward_scale

def train_from_torch(self, batch):
    gt.blank_stamp()
    pass

def compute_loss(
    self, 
    batch,
    skip_statistics=False
):
    # Transitions
    rewards = batch["rewards"]
    obs = batch["observations"]
    actions = batch["actions"]
    next_obs = batch["next_observations"]

def train_networks(
    self,
    batch,
    skip_statistics=False
)-> LossStatistics:
    #TODO Figure out how to deal with the batches
    #TODO Call the loss function on the returns
    #TODO Backprop on the returns
    pass