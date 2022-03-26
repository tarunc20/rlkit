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
        self.valid_buffer = valid_buffer

    def train_from_torch(self, batch):
        self.policy_optimizer.zero_grad(set_to_none=True)
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self._n_train_steps_total += 1

        self.eval_statistics = stats

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
        action_preds = self.policy(obs)
        loss = self.policy_criterion(action_preds, actions)
        eval_statistics = OrderedDict()
        eval_statistics["Policy MSE"] = self.policy_criterion(
            action_preds, actions
        ).item()
        eval_statistics["Predicted Actions Abs Max"] = action_preds.abs().max().item()
        eval_statistics["Predicted Actions Abs Mean"] = action_preds.abs().mean().item()
        eval_statistics["Predicted Actions Mean"] = action_preds.mean().item()
        eval_statistics["Predicted Actions Std"] = action_preds.std().item()
        eval_statistics["Actions Abs Max"] = actions.abs().max().item()
        eval_statistics["Actions Abs Mean"] = actions.abs().mean().item()
        eval_statistics["Actions Mean"] = actions.mean().item()
        eval_statistics["Actions Std"] = actions.std().item()
        eval_statistics["High Level Actions Std"] = (
            obs[:, 64 * 64 * 3 + 8 : -1].std().item()
        )

        print(self._n_train_steps_total)
        print(f"Policy MSE: {eval_statistics['Policy MSE']}")
        print(f"Predicted Actions Abs Max {action_preds.abs().max().item()}")
        print(f"Predicted Actions Abs Mean {action_preds.abs().mean().item()}")
        print(f"Predicted Actions Mean {action_preds.mean().item()}")
        print(f"Predicted Actions Std {action_preds.std().item()}")

        print(f"Actions Abs Max {actions.abs().max().item()}")
        print(f"Actions Abs Mean {actions.abs().mean().item()}")
        print(f"Actions Mean {actions.mean().item()}")
        print(f"Actions Std {actions.std().item()}")
        print(f"High level action std {obs[:, 64*64*3+8:-1].std().item()}")
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
                eval_statistics["Valid Predicted Actions Abs Max"] = (
                    valid_action_preds.abs().max().item()
                )
                eval_statistics["Valid Predicted Actions Abs Mean"] = (
                    valid_action_preds.abs().mean().item()
                )
                eval_statistics[
                    "Valid Predicted Actions Mean"
                ] = valid_action_preds.mean().item()
                eval_statistics["Valid Predicted Actions Abs Max"] = (
                    valid_action_preds.abs().max().item()
                )
                eval_statistics[
                    "Valid Actions Abs Mean"
                ] = valid_actions.abs.mean().item()
                eval_statistics["Valid Actions Mean"] = valid_actions.mean().item()

                print(f"Valid Policy MSE: {eval_statistics['Valid Policy MSE']}")
                print(
                    f"Valid Predicted Actions Abs Max {valid_action_preds.abs().max().item()}"
                )
                print(
                    f"Valid Predicted Actions Abs Mean {valid_action_preds.abs().mean().item()}"
                )
                print(
                    f"Valid Predicted Actions Mean {valid_action_preds.mean().item()}"
                )

                print(f"Valid Actions Abs Max {valid_actions.abs().max().item()}")
                print(f"Valid Actions Abs Mean {valid_actions.abs().mean().item()}")
                print(f"Valid Actions Mean {valid_actions.mean().item()}")
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
