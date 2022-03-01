import torch
import torch.nn.functional as F
from torch import jit
from torch import nn as nn
from torch.distributions import Normal, TransformedDistribution

import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.torch.distributions import (
    MultivariateDiagonalNormal,
    TanhNormal,
    TorchDistributionWrapper,
)
from rlkit.torch.model_based.dreamer.actor_models import (
    Independent,
    SafeTruncatedNormal,
    SampleDist,
    TanhBijector,
)
from rlkit.torch.model_based.dreamer.mlp import Mlp


class CNN(jit.ScriptModule):
    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        kernel_sizes,
        n_channels,
        strides,
        paddings,
        hidden_sizes=None,
        hidden_init=nn.init.xavier_uniform_,
        hidden_activation=nn.ReLU,
        output_activation=identity,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_input_length = (
            self.input_width * self.input_height * self.input_channels
        )

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                hidden_init(m.weight)
                m.bias.data.fill_(0)

        self.conv_block_1 = nn.Sequential(
            *[
                nn.Conv2d(
                    self.input_channels,
                    n_channels[0],
                    kernel_sizes[0],
                    stride=strides[0],
                    padding=paddings[0],
                ),
                hidden_activation(inplace=True),
            ]
        )
        input_channels = n_channels[0]
        self.conv_block_1.apply(init_weights)

        conv_block_2 = []

        for out_channels, kernel_size, stride, padding in zip(
            n_channels[1:], kernel_sizes[1:], strides[1:], paddings[1:]
        ):
            conv = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            conv_block_2.append(conv_layer)
            conv_block_2.append(hidden_activation(inplace=True))
            input_channels = out_channels
        self.conv_block_2 = nn.Sequential(*conv_block_2)
        self.to(memory_format=torch.channels_last)

    @jit.script_method
    def forward(self, input_):
        conv_input = input_.narrow(
            start=0, length=self.conv_input_length, dim=1
        ).contiguous()
        h = conv_input.view(
            conv_input.shape[0],
            self.input_channels,
            self.input_height,
            self.input_width,
        ).to(
            memory_format=torch.channels_last, device=ptu.device, dtype=conv_input.dtype
        )

        h = self.conv_block_1(h)
        h = self.conv_block_2(h)
        output = h.reshape(h.size(0), -1)
        return output


class DCNN(jit.ScriptModule):
    def __init__(
        self,
        fc_input_size,
        hidden_sizes,
        deconv_input_width,
        deconv_input_height,
        deconv_input_channels,
        deconv_output_kernel_size,
        deconv_output_strides,
        deconv_output_channels,
        kernel_sizes,
        n_channels,
        strides,
        paddings,
        hidden_init=nn.init.xavier_uniform_,
        hidden_activation=nn.ReLU(),
        output_activation=identity,
    ):
        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = (
            self.deconv_input_channels
            * self.deconv_input_height
            * self.deconv_input_width
        )

        fc = nn.Linear(fc_input_size, deconv_input_size)
        hidden_init(fc.weight)
        fc.bias.data.fill_(0)
        self.fc_block = nn.Sequential(fc, hidden_activation(inplace=True))

        deconv_layers = []

        for out_channels, kernel_size, stride, padding in zip(
            n_channels, kernel_sizes, strides, paddings
        ):
            deconv = nn.ConvTranspose2d(
                deconv_input_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
            hidden_init(deconv.weight)
            deconv.bias.data.fill_(0)

            deconv_layer = deconv
            deconv_layers.append(deconv_layer)
            deconv_layers.append(nn.ReLU(inplace=True))
            deconv_input_channels = out_channels
        deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(deconv_output.weight)
        deconv_output.bias.data.fill_(0)
        deconv_layers.append(deconv_output)
        self.deconv_block = nn.Sequential(*deconv_layers)
        self.to(memory_format=torch.channels_last)

    @jit.script_method
    def forward(self, input_):
        h = self.fc_block(input_)
        h = h.view(
            -1,
            self.deconv_input_channels,
            self.deconv_input_width,
            self.deconv_input_height,
        ).to(memory_format=torch.channels_last, device=ptu.device, dtype=h.dtype)
        output = self.deconv_block(h)
        return output.to(
            memory_format=torch.channels_last, device=ptu.device, dtype=h.dtype
        )


class CNNMLP(jit.ScriptModule):
    def __init__(
        self,
        image_encoder_args,
        image_encoder_kwargs,
        state_encoder_args,
        state_encoder_kwargs,
        joint_processor_args,
        joint_processor_kwargs,
        image_dim,
        mean_scale=15,
        dist="normal",
    ):
        super().__init__()
        self.image_encoder = CNN(*image_encoder_args, **image_encoder_kwargs)
        self.state_encoder = Mlp(*state_encoder_args, **state_encoder_kwargs)
        self.joint_processor = Mlp(*joint_processor_args, **joint_processor_kwargs)
        self.image_dim = image_dim

    @jit.script_method
    def forward(self, input_):
        image, state = input_[:, : self.image_dim], input_[:, self.image_dim :]
        image = self.preprocess(image)
        image_encoding = self.image_encoder(image)
        state_encoding = self.state_encoder(state)
        joint_encoding = torch.cat((image_encoding, state_encoding), dim=1)
        output = self.joint_processor(joint_encoding)
        return output

    @jit.script_method
    def preprocess(self, obs):
        assert (
            obs.max() <= 255.0 and obs.min() >= 0.0
        ), f"Invalid observation: max: {obs.max()}, min: {obs.min()}"
        obs = obs / 255.0 - 0.5
        return obs


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class CNNMLPGaussian(jit.ScriptModule):
    """
    Usage:

    ```
    policy = TanhCNNMLPGaussianPolicy(...)
    assumes joint processor outputs 2*output_size
    """

    def __init__(
        self,
        image_encoder_args,
        image_encoder_kwargs,
        state_encoder_args,
        state_encoder_kwargs,
        joint_processor_args,
        joint_processor_kwargs,
        image_dim,
        min_std=0.1,
        init_std=0.0,
        mean_scale=15,
        dist="normal",
    ):
        super().__init__()
        self.image_encoder = CNN(*image_encoder_args, **image_encoder_kwargs)
        self.state_encoder = Mlp(*state_encoder_args, **state_encoder_kwargs)
        self.joint_processor = Mlp(*joint_processor_args, **joint_processor_kwargs)
        self.image_dim = image_dim
        self.raw_init_std = torch.log(torch.exp(ptu.tensor(init_std)) - 1)
        self._min_std = min_std
        self._mean_scale = mean_scale
        self._dist = dist

    @jit.script_method
    def preprocess(self, obs):
        assert (
            obs.max() <= 255.0 and obs.min() >= 0.0
        ), f"Invalid observation: max: {obs.max()}, min: {obs.min()}"
        obs = obs / 255.0 - 0.5
        return obs

    @jit.script_method
    def forward_net(self, input_):
        image, state = input_[:, : self.image_dim], input_[:, self.image_dim :]
        image = self.preprocess(image)
        image_encoding = self.image_encoder(image)
        state_encoding = self.state_encoder(state)
        joint_encoding = torch.cat((image_encoding, state_encoding), dim=1)
        output = self.joint_processor(joint_encoding)
        mean, std = output.split(self.joint_processor.output_size // 2, dim=-1)
        return mean, std

    def forward(self, input_):
        mean, std = self.forward_net(input_)
        if self._dist == "tanh_normal_dreamer_v1":
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + self.raw_init_std) + self._min_std
            assert (
                std >= 0.0
            ).all(), f"std should not be negative, {std.max(), std.min()}"
            dist = Normal(mean, std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = Independent(dist, 1)
            dist = SampleDist(dist)
            dist = TorchDistributionWrapper(dist)
        elif self._dist == "trunc_normal":
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            assert (
                std >= 0.0
            ).all(), f"std should not be negative, {std.max(), std.min()}"
            dist = SafeTruncatedNormal(
                mean, std, -self._mean_scale, self._mean_scale, mult=self._mean_scale
            )
            dist = Independent(dist, 1)
            dist = TorchDistributionWrapper(dist)
        elif self._dist == "normal":
            std = F.softplus(std + self.raw_init_std) + self._min_std
            assert (
                std >= 0.0
            ).all(), f"std should not be negative, {std.max(), std.min()}"
            dist = MultivariateDiagonalNormal(mean, std)
        elif self._dist == "tanh_normal":
            log_std = torch.clamp(std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
            assert (
                std >= 0.0
            ).all(), f"std should not be negative, {std.max(), std.min()}"
            dist = TanhNormal(mean, std)
        return dist
