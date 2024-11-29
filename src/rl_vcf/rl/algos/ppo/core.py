import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, Space
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    """Initialize layer weights."""
    # Note: different to standard torch initialization methods
    # Used by the original PPO implementation
    torch.nn.init.orthogonal_(layer.weight, std)  # outperforms default init
    torch.nn.init.constant_(layer.bias, bias_const)  # start with 0 bias
    return layer


def mlp(
    sizes: tuple[int],
    activation: nn.Module,
    output_activation: nn.Module = nn.Identity(),
    output_std: float = np.sqrt(2),
) -> nn.Sequential:
    """Create a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [
                layer_init(nn.Linear(sizes[j], sizes[j + 1]), std=np.sqrt(2)),
                activation,
            ]
        else:
            layers += [
                layer_init(nn.Linear(sizes[j], sizes[j + 1]), std=output_std),
                output_activation,
            ]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Actor base class."""

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

    def _log_prob_from_distribution(
        self, pi: torch.distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.distributions.Distribution, torch.Tensor | None]:
        # Produce action distribution for given observations, and optionally compute the
        # log likelihood of given actions under those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    """Categorical actor."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int],
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation, output_std=0.01
        )

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(
        self, pi: Categorical, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act.long())


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MLPGaussianActor(Actor):
    """Gaussian actor."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int],
        activation: nn.Module,
    ) -> None:
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation, output_std=0.01
        )

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(
        self, pi: Normal, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    """Critic."""

    def __init__(self, obs_dim, hidden_sizes, activation) -> None:
        super().__init__()
        self.v_net = mlp(
            [obs_dim] + list(hidden_sizes) + [1], activation, output_std=1.0
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    """Actor and critic."""

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        hidden_sizes: tuple[int] = (64, 64),
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()

        obs_dim = np.array(observation_space.shape).prod()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, np.prod(action_space.shape), hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value from critic."""
        return self.v(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (and log-prob) from actor and get value from critic."""
        pi = self.pi._distribution(obs)
        if act is None:
            act = pi.sample()
        return (
            act,
            self.pi._log_prob_from_distribution(pi, act),
            pi.entropy(),
            self.v(obs),
        )

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample action from actor."""
        return self.get_action_and_value(obs)[0]


# From OpenAI SpinningUp, may be useful later:


def combined_shape(length: int, shape: int | tuple[int] | None = None) -> tuple[int]:
    """Create a tuple of size (length, shape)."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: nn.Module) -> int:
    """Count the number of parameters in a module."""
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute discounted cumulative sums of vectors.

    input:
    array x:
        [x0,
        x1,
        ...
        xN]

    output:
        [x0 + discount * x1 + ... + discount^N * xN,
        x1 + discount * x2 + ... + discount^(N-1) * xN,
        ...
        xN]

    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
