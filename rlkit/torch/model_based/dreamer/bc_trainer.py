import os
from collections import OrderedDict, namedtuple
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
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
        valid_buffer=None,
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
        self.valid_buffer = valid_buffer

    def train_from_torch(self, batch):
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        # self.scaler.scale(losses.policy_loss).backward()
        # update_network(self.policy, self.policy_optimizer, 0, self.scaler)
        # self.scaler.update()

        self._n_train_steps_total += 1

        self.eval_statistics = stats

    @torch.cuda.amp.autocast()
    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[BCLosses, LossStatistics]:
        obs = batch["observations"]
        actions = batch["actions"]
        import ipdb

        ipdb.set_trace()
        """
        Policy Loss
        """
        # action_dist = self.policy.get_dist(
        #     obs, torch.zeros_like(obs), torch.zeros_like(obs)
        # )
        # loss = -1 * action_dist.log_prob(actions).mean()
        action_preds = self.policy(obs)
        print((action_preds == actions).all())
        loss = self.policy_criterion(action_preds, actions)
        eval_statistics = OrderedDict()
        eval_statistics["Policy MSE"] = self.policy_criterion(
            action_preds, actions
        ).item()
        eval_statistics["Policy Loss"] = loss.item()
        eval_statistics["Predicted Actions Mean"] = action_preds.mean().item()
        eval_statistics["Predicted Actions Max"] = action_preds.abs().max().item()
        eval_statistics["Actions Mean"] = actions.mean().item()

        print(self._n_train_steps_total)
        print(f"Policy Loss: {loss.item()}")
        print(f"Policy MSE: {eval_statistics['Policy MSE']}")
        print(f"Predicted Actions Max {action_preds.abs().max().item()}")
        print(f"Actions Max {actions.abs().max().item()}")
        print()

        with torch.no_grad():
            if self.valid_buffer:
                valid_batch = self.valid_buffer.random_batch(actions.shape[0] * 16)
                valid_obs = ptu.from_numpy(valid_batch["observations"])
                valid_actions = ptu.from_numpy(valid_batch["actions"])
                valid_action_preds = self.policy(valid_obs)
                valid_loss = self.policy_criterion(valid_action_preds, valid_actions)
                eval_statistics["Valid Policy MSE"] = self.policy_criterion(
                    valid_action_preds, valid_actions
                ).item()
                eval_statistics["Valid Policy Loss"] = valid_loss.item()
                eval_statistics[
                    "Valid Predicted Actions Mean"
                ] = valid_action_preds.mean().item()
                eval_statistics["Valid Predicted Actions Mean"] = (
                    valid_action_preds.abs().mean().item()
                )
                eval_statistics["Valid Actions Mean"] = valid_actions.mean().item()
                eval_statistics["Valid Predicted Actions Max"] = (
                    valid_action_preds.abs().max().item()
                )

                print(f"Valid Policy Loss: {valid_loss.item()}")
                print(f"Valid Policy MSE: {eval_statistics['Valid Policy MSE']}")
                print(
                    f"Valid Predicted Actions Max {valid_action_preds.abs().max().item()}"
                )
                print(f"Valid Actions Max {valid_actions.abs().max().item()}")
                print()

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
