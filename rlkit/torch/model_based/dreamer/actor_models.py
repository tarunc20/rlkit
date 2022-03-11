import torch
import torch.nn.functional as F
from torch import jit
from torch.distributions import Normal, Transform, TransformedDistribution
from torch.distributions.one_hot_categorical import OneHotCategorical

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.truncated_normal import TruncatedNormal


class ActorModel(Mlp):
    def __init__(
        self,
        hidden_size,
        obs_dim,
        num_layers=4,
        discrete_continuous_dist=False,
        discrete_action_dim=0,
        continuous_action_dim=0,
        hidden_activation=F.elu,
        min_std=0.1,
        init_std=0.0,
        mean_scale=5.0,
        use_tanh_normal=True,
        dist="trunc_normal",
        **kwargs,
    ):
        self.discrete_continuous_dist = discrete_continuous_dist
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        if self.discrete_continuous_dist:
            self.output_size = self.discrete_action_dim + self.continuous_action_dim * 2
        else:
            self.output_size = self.continuous_action_dim * 2
        super().__init__(
            [hidden_size] * num_layers,
            input_size=obs_dim,
            output_size=self.output_size,
            hidden_activation=hidden_activation,
            hidden_init=torch.nn.init.xavier_uniform_,
            **kwargs,
        )
        self._min_std = min_std
        self._mean_scale = mean_scale
        self.use_tanh_normal = use_tanh_normal
        self._dist = dist
        self.raw_init_std = torch.log(torch.exp(ptu.tensor(init_std)) - 1)

    @jit.script_method
    def forward_net(self, input_):
        h = input_
        if self.apply_embedding:
            embed_h = h[:, : self.embedding_slice]
            embedding = self.embedding(embed_h.argmax(dim=1))
            h = torch.cat([embedding, h[:, self.embedding_slice :]], dim=1)
        h = self.fc_block_1(h)
        preactivation = self.fc_block_2(h)
        output = preactivation
        return output

    def get_continuous_dist(self, mean, std):
        if self._dist == "tanh_normal_dreamer_v1":
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + self.raw_init_std) + self._min_std
            dist = Normal(mean, std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = Independent(dist, 1)
        return dist

    def forward(self, input_):
        last = self.forward_net(input_)
        if self.discrete_continuous_dist:
            assert last.shape[1] == self.output_size
            mean, continuous_action_std = (
                last[:, : self.discrete_action_dim + self.continuous_action_dim],
                last[:, self.discrete_action_dim + self.continuous_action_dim :],
            )
            split = mean.split(self.discrete_action_dim, -1)
            if len(split) == 2:
                discrete_logits, continuous_action_mean = split
            else:
                discrete_logits, continuous_action_mean, extra = split
                continuous_action_mean = torch.cat((continuous_action_mean, extra), -1)
            dist1 = OneHotDist(logits=discrete_logits)
            dist2 = self.get_continuous_dist(
                continuous_action_mean, continuous_action_std
            )
            dist = SplitDist(dist1, dist2, self.discrete_action_dim)
        else:
            action_mean, action_std = last.split(self.continuous_action_dim, -1)
            dist = self.get_continuous_dist(action_mean, action_std)
        return dist

    @jit.script_method
    def compute_exploration_action(self, action, expl_amount: float):
        if expl_amount == 0:
            return action
        else:
            if self.discrete_continuous_dist:
                discrete, continuous = (
                    action[:, : self.discrete_action_dim],
                    action[:, self.discrete_action_dim :],
                )
                indices = torch.randint(
                    0, discrete.shape[-1], discrete.shape[0:-1], device=ptu.device
                ).long()
                rand_action = F.one_hot(indices, discrete.shape[-1])
                probs = torch.rand(discrete.shape[:1], device=ptu.device)
                discrete = torch.where(
                    probs.reshape(-1, 1) < expl_amount,
                    rand_action.int(),
                    discrete.int(),
                )
                continuous = torch.normal(continuous, expl_amount)
                if self.use_tanh_normal:
                    continuous = torch.clamp(continuous, -1, 1)
                action = torch.cat((discrete, continuous), -1)
            else:
                action = torch.normal(action, expl_amount)
                if self.use_tanh_normal:
                    action = torch.clamp(action, -1, 1)
            return action


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
@torch.jit.script
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (torch.log(ptu.tensor(2.0)) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        mean = torch.mean(sample, 0)
        return mean

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def log_prob(self, actions):
        return self._dist.log_prob(actions)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def rsample(self):
        return self._dist.rsample()


class OneHotDist(OneHotCategorical):
    def mode(self):
        return self._one_hot(torch.argmax(self.probs, dim=-1))

    def rsample(self, sample_shape=torch.Size()):
        sample = super().sample(sample_shape)
        probs = self.probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - (probs).detach()  # straight through estimator
        return sample

    def _one_hot(self, indices):
        return F.one_hot(indices, self._categorical._num_events)


class SplitDist:
    def __init__(self, dist1, dist2, split_dim):
        self._dist1 = dist1
        self._dist2 = dist2
        self.split_dim = split_dim

    def rsample(self):
        return torch.cat((self._dist1.rsample(), self._dist2.rsample()), -1)

    def mode(self):
        return torch.cat((self._dist1.mode().float(), self._dist2.mode().float()), -1)

    def entropy(self):
        return self._dist1.entropy() + self._dist2.entropy()

    def log_prob(self, actions):
        return self._dist1.log_prob(
            actions[:, : self.split_dim]
        ) + self._dist2.log_prob(actions[:, self.split_dim :])


class SafeTruncatedNormal(TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

    def rsample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        clipped = torch.max(
            torch.min(event, ptu.ones_like(event) - self._clip),
            -1 * ptu.ones_like(event) + self._clip,
        )
        event = event - event.detach() + clipped.detach()
        event *= self._mult
        return event

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SafeTruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        new._clip = self._clip
        new._mult = self._mult
        super(SafeTruncatedNormal, new).__init__(
            new.loc, new.scale, new.a, new.b, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


class Independent(torch.distributions.Independent):
    def mode(self):
        return self.base_dist.mode()


class TanhNormalGarage(torch.distributions.Distribution):
    r"""A distribution induced by applying a tanh transformation to a Gaussian random variable.
    Algorithms like SAC and Pearl use this transformed distribution.
    It can be thought of as a distribution of X where
        :math:`Y ~ \mathcal{N}(\mu, \sigma)`
        :math:`X = tanh(Y)`
    Args:
        loc (torch.Tensor): The mean of this distribution.
        scale (torch.Tensor): The stdev of this distribution.
    """  # noqa: 501

    def __init__(self, loc, scale):
        self._normal = Independent(Normal(loc, scale), 1)
        super().__init__()

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
        """The log likelihood of a sample on the this Tanh Distribution.
        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                function applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Regularization constant. Making this value larger
                makes the computation more stable but less precise.
        Note:
              when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`
        Returns:
            torch.Tensor: The log likelihood of value on the distribution.
        """
        # pylint: disable=arguments-differ
        if pre_tanh_value is None:
            pre_tanh_value = (
                torch.log((1 + epsilon + value) / (1 + epsilon - value)) / 2
            )
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = norm_lp - torch.sum(
            torch.log(self._clip_but_pass_gradient((1.0 - value ** 2)) + epsilon),
            axis=-1,
        )
        return ret

    def sample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.
        Args:
            sample_shape (list): Shape of the returned value.
        Note:
            Gradients `do not` pass through this operation.
        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.
        Args:
            sample_shape (list): Shape of the returned value.
        Note:
            Gradients pass through this operation.
        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.
        """
        z = self._normal.rsample(sample_shape)
        return torch.tanh(z)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal distribution.
        Returns the sampled value before the tanh transform is applied and the
        sampled value with the tanh transform applied to it.
        Args:
            sample_shape (list): shape of the return.
        Note:
            Gradients pass through this operation.
        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed with `tanh`.
        """
        z = self._normal.rsample(sample_shape)
        return z, torch.tanh(z)

    def cdf(self, value):
        """Returns the CDF at the value.
        Returns the cumulative density/mass function evaluated at
        `value` on the underlying normal distribution.
        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.
        Returns:
            torch.Tensor: the result of the cdf being computed.
        """
        return self._normal.cdf(value)

    def icdf(self, value):
        """Returns the icdf function evaluated at `value`.
        Returns the icdf function evaluated at `value` on the underlying
        normal distribution.
        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.
        Returns:
            torch.Tensor: the result of the cdf being computed.
        """
        return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
        """Construct a new TanhNormal distribution from a normal distribution.
        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new TanhNormal distribution.
        Returns:
            TanhNormal: A new distribution whose underlying normal dist
                is new_normal.
        """
        # pylint: disable=protected-access
        new = cls(torch.zeros(1), torch.zeros(1))
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new TanhNormal distribution.
        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.
        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.
        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.
        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal)
        return new

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete dist.
        The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).
        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.
        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.
        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.
        Note:
            Calls the enumerate_support function of the underlying normal
            distribution.
        Returns:
            torch.Tensor: Tensor iterating over dimension 0.
        """
        return self._normal.enumerate_support(expand)

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self._normal.mean)

    @property
    def variance(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self._normal.variance

    def entropy(self):
        """Returns entropy of the underlying normal distribution.
        Returns:
            torch.Tensor: entropy of the underlying normal distribution.
        """
        return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0.0, upper=1.0):
        """Clipping function that allows for gradients to flow through.
        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping
        Returns:
            torch.Tensor: x clipped between lower and upper.
        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = (upper - x) * clip_up + (lower - x) * clip_low
        return x + clip

    def __repr__(self):
        """Returns the parameterization of the distribution.
        Returns:
            str: The parameterization of the distribution and underlying
                distribution.
        """
        return self.__class__.__name__
