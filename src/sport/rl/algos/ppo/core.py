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

    def forward(
        self,
        obs: torch.Tensor,
        pi_action: torch.Tensor | None = None,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        logits = self.logits_net(obs)

        # Sample
        pi_distribution = Categorical(logits=logits)
        if pi_action is None:
            if deterministic:
                pi_action = pi_distribution.mode
            else:
                pi_action = pi_distribution.sample()

        if with_log_prob:
            logp_pi = pi_distribution.log_prob(pi_action.long())
        else:
            logp_pi = None

        return pi_action, logp_pi, pi_distribution.entropy(), None


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

    def forward(
        self,
        obs: torch.Tensor,
        pi_action: torch.Tensor | None = None,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)

        # Sample
        pi_distribution = Normal(mu, std)
        if pi_action is None:
            if deterministic:
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

        if with_log_prob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Last axis sum needed for Torch Normal distribution
        else:
            logp_pi = None

        return pi_action, logp_pi, pi_distribution.entropy(), None


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MLPSquashedGaussianActor(nn.Module):
    """Squashed Gaussian actor."""

    # This actor has a state-dependent standard deviation, and is designed to match the
    # actor from SAC (so we do stuff like transforming the action).
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int],
        activation: nn.Module,
        act_scale: torch.Tensor,
        act_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.register_buffer("act_scale", act_scale)
        self.register_buffer("act_bias", act_bias)
        # Save activation function as a string in the register buffer
        self.register_buffer("activation", torch.tensor([0]))
        ords = list(map(ord, activation.__class__.__name__))
        self.activation = torch.tensor(ords)

    def get_activation(self) -> str:
        # Get the activation function as a string from the register buffer
        ords = self.activation.tolist()
        return "".join(map(chr, ords))

    def forward(
        self,
        obs: torch.Tensor,
        pi_action: torch.Tensor | None = None,  # this must be untransformed action x_t
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        x_t = pi_action
        if x_t is None:
            if deterministic:
                x_t = mu
            else:
                x_t = pi_distribution.rsample()
        y_t = torch.tanh(x_t)
        pi_action = y_t * self.act_scale + self.act_bias

        if with_log_prob:
            logp_pi = pi_distribution.log_prob(x_t)
            # Enforcing action bounds
            logp_pi -= torch.log(self.act_scale * (1 - y_t.pow(2)) + 1e-6)
            logp_pi = logp_pi.sum(1, keepdim=True).squeeze(-1)
            # why doesn't cleanrl squeeze?
        else:
            logp_pi = None

        return pi_action, logp_pi, pi_distribution.entropy(), x_t


class MLPCritic(nn.Module):
    """Critic."""

    def __init__(self, obs_dim, hidden_sizes, activation) -> None:
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    """Actor and critic."""

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        hidden_sizes: tuple[int] = (64, 64),
        activation: nn.Module = nn.Tanh(),
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        obs_dim = np.array(observation_space.shape).prod()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            act_dim = np.prod(action_space.shape)
            if state_dependent_std:
                # Some envs define the action space dims as a double, so we need to force float
                act_scale = torch.tensor(
                    (action_space.high - action_space.low) / 2.0, dtype=torch.float32
                )
                act_bias = torch.tensor(
                    (action_space.high + action_space.low) / 2.0, dtype=torch.float32
                )
                self.pi = MLPSquashedGaussianActor(
                    obs_dim,
                    act_dim,
                    hidden_sizes,
                    activation,
                    act_scale,
                    act_bias,
                )
            else:
                self.pi = MLPGaussianActor(
                    obs_dim,
                    act_dim,
                    hidden_sizes,
                    activation,
                )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        # Note we use V(s) as the value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value from critic."""
        return self.v.forward(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Sample action (and log-prob) from actor and get value from critic."""
        act, logp_a, entropy, x_t = self.pi.forward(
            obs, act, deterministic=False, with_log_prob=True
        )
        return (
            act,
            logp_a,
            entropy,
            self.v.forward(obs),
            x_t,  # Note: this is None unless we use state-dependent std
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from actor."""
        return self.pi.forward(obs, None, deterministic, False)[0]


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
