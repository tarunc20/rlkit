from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.loss import LossFunction, LossStatistics

import torch
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
            self.policy.parameters(),
            lr=policy_lr
        )

        self._need_to_update_eval_statistics = False
        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0

    def train_from_torch(self, batch):
        gt.blank_stamp()
        
        stats = self.train_networks(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics
        )

        self._n_train_steps_total += 1
        
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False

    @property
    def networks(self):
        return [self.policy]

    @property
    def optimizers(self):
        return [self.policy_optimizer]

    def compute_loss(
        self, 
        batch,
        skip_statistics=False
    ):
        # Transitions
        rewards = batch["rewards"]
        obs = batch["observations"]
        actions = batch["actions"]

        dist = self.policy(obs.reshape(-1, obs.shape[-1]))
        log_prob = dist.log_prob(actions.reshape(-1, actions.shape[-1]))
        batch_weight = torch.sum(rewards)

        return -(log_prob * batch_weight).mean()

    def train_networks(
        self,
        batch,
        skip_statistics=False
    )-> LossStatistics:
        self.policy_optimizer.zero_grad()
        batch_loss = self.compute_loss(batch, False)
        batch_loss.backward()
        self.policy_optimizer.step()

        return None