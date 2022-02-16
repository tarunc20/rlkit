from collections import OrderedDict, namedtuple
from typing import Tuple

import gtimer as gt
import torch
import torch.optim as optim
from torch import nn as nn

from rlkit.core.loss import LossFunction, LossStatistics
from rlkit.torch.model_based.dreamer.utils import update_network
from rlkit.torch.torch_rl_algorithm import TorchTrainer

BCLosses = namedtuple(
    "BCLosses",
    "policy_loss",
)


class BCTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        policy,
        policy_lr=1e-3,
        optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.policy = policy
        self.policy_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self.scaler = torch.cuda.amp.GradScaler()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        self.scaler.scale(losses.policy_loss).backward()
        update_network(self.policy, self.policy_optimizer, 0, self.scaler)
        self.scaler.update()

        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp("bc training", unique=False)

    @torch.cuda.amp.autocast()
    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[BCLosses, LossStatistics]:
        obs = batch["observations"]
        actions = batch["actions"]

        """
        Policy Loss
        """
        action_preds = self.policy(obs.reshape(-1, obs.shape[-1]))
        loss = self.policy_criterion(
            action_preds, actions.reshape(-1, actions.shape[-1])
        )
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["Policy Loss"] = loss.item()

        loss = BCLosses(
            policy_loss=loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
        ]

    @property
    def optimizers(self):
        return [
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )
