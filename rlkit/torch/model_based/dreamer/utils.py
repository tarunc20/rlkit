import re

import numpy as np
import torch


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
class FreezeParameters:
    def __init__(self, params):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because
        gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
                        output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.params = params
        self.param_states = [param.requires_grad for param in params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


@torch.jit.script
def lambda_return(reward, value, discount, bootstrap, lambda_: float = 0.95):
    # from: https://github.com/yusukeurakami/dreamer-pytorch
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    """
    Compute the discounted reward for a batch of data.
    arguments:
        reward: [horizon - 1, batch, 1]
        value: [horizon - 1, batch, 1]
        discount: [horizon - 1, batch, 1]
        bootstrap: [batch, 1]
    returns:
        returns: [horizon - 1, batch, 1]
    """
    assert reward.shape[0] == value.shape[0] == discount.shape[0]
    assert reward.shape[1] == value.shape[1] == discount.shape[1]
    assert reward.shape[1] == bootstrap.shape[0]
    assert reward.shape[0] > 0
    next_values = torch.cat([value[1:], bootstrap.unsqueeze(0)], 0)
    target = reward + discount * next_values * (1 - lambda_)
    outputs = []
    accumulated_reward = bootstrap
    for t in range(reward.shape[0] - 1, -1, -1):
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


# from dreamer_v2 repo
def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = np.clip(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = np.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        raise NotImplementedError(string)
