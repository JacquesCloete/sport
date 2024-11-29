import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.distributions import Normal
from torch.nn.functional import softplus


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        size: int,
        device: torch.device,
    ) -> None:
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))
        self.obs_buf = torch.zeros((size, obs_dim)).to(device)
        self.obs2_buf = torch.zeros((size, obs_dim)).to(device)
        self.act_buf = torch.zeros((size, act_dim)).to(device)
        self.rew_buf = torch.zeros(size).to(device)
        self.done_buf = torch.zeros(size).to(device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )


def mlp(
    sizes: tuple[int],
    activation: nn.Module,
    output_activation: nn.Module = nn.Identity(),
) -> nn.Sequential:
    """Create a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [
                nn.Linear(sizes[j], sizes[j + 1]),
                activation,
            ]
        else:
            layers += [
                nn.Linear(sizes[j], sizes[j + 1]),
                output_activation,
            ]
    return nn.Sequential(*layers)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MLPSquashedGaussianActor(nn.Module):
    """Squashed Gaussian actor."""

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

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False, with_log_prob: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_log_prob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = pi_action * self.act_scale + self.act_bias

        return pi_action, logp_pi


class MLPCritic(nn.Module):
    """Critic."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int],
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.q_net = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            activation,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(
            self.q_net(torch.cat([obs, act], dim=-1)), -1
        )  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    """Actor and critic."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_sizes: tuple[int] = (256, 256),
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        obs_dim = np.array(observation_space.shape).prod()
        act_dim = np.prod(action_space.shape)
        act_scale = torch.tensor(
            (action_space.high - action_space.low) / 2.0, dtype=torch.float32
        )
        act_bias = torch.tensor(
            (action_space.high + action_space.low) / 2.0, dtype=torch.float32
        )

        # Build policy and value functions
        self.pi = MLPSquashedGaussianActor(
            obs_dim, act_dim, hidden_sizes, activation, act_scale, act_bias
        )
        self.q1 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            """Sample action from actor."""
            return self.pi(obs, deterministic, False)[0]


# From OpenAI SpinningUp, may be useful later:


def combined_shape(length: int, shape: int | tuple[int] | None = None) -> tuple[int]:
    """Create a tuple of size (length, shape)."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: nn.Module) -> int:
    """Count the number of parameters in a module."""
    return sum([np.prod(p.shape) for p in module.parameters()])
