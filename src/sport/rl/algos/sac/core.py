import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.distributions import Normal


class ReplayBuffer:
    """
    A FIFO experience replay buffer for SAC agents, with the option for prioritization.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        size: int,
        device: torch.device,
        prioritized: bool = False,
        alpha: float = 0.6,
    ) -> None:
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))
        self.obs_buf = torch.zeros((size, obs_dim), requires_grad=False).to(device)
        self.obs2_buf = torch.zeros((size, obs_dim), requires_grad=False).to(device)
        self.act_buf = torch.zeros((size, act_dim), requires_grad=False).to(device)
        self.rew_buf = torch.zeros(size, requires_grad=False).to(device)
        self.done_buf = torch.zeros(size, requires_grad=False).to(device)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.prioritized = prioritized
        if self.prioritized:
            self.alpha = alpha
            self.priorities = torch.ones(size, requires_grad=False).to(device)

    def store(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if self.prioritized:
            # New transitions have max probability
            self.priorities[self.ptr] = torch.max(self.priorities)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(
        self, batch_size: int, beta: float = 0.4
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        if self.prioritized:
            scaled_priorities = self.priorities[: self.size] ** self.alpha
            sampling_probs = scaled_priorities / torch.sum(scaled_priorities)
            idxs = torch.multinomial(
                input=sampling_probs,
                num_samples=batch_size,
                replacement=True,
            )
            weights = (self.size * sampling_probs[idxs]) ** (-beta)
            weights = weights / torch.max(weights)  # Normalize for stability
            weights.requires_grad = False
        else:
            idxs = torch.randint(0, self.size, size=(batch_size,))
            weights = torch.ones(self.size, requires_grad=False)
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
            idx=idxs,
            wgt=weights,
        )

    def update_priorities(self, idxs: torch.Tensor, td_errors: torch.Tensor) -> None:
        """Update priorities based on TD errors."""
        self.priorities[idxs] = (
            torch.abs(td_errors) + 1e-6
        )  # add small epsilon for stability


class TaskSuccessBuffer:
    """
    A FIFO buffer to record task success during training.
    Used for curriculum learning.
    """

    def __init__(
        self,
        size: int,
    ) -> None:
        self.buf = np.full(size, False, dtype=bool)
        self.max_size = size
        self.reset()

    def store(
        self,
        data: bool,
    ) -> None:
        """Add new data to the replay buffer."""
        self.buf[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_success_rate(self) -> float:
        """Get the empirical task success rate."""
        assert self.size > 0, "Task success buffer is empty."
        return np.sum(self.buf) / self.size

    def reset(self) -> None:
        """Reset the buffer."""
        self.buf = np.full(self.max_size, False, dtype=bool)
        self.ptr, self.size = 0, 0


def layer_init(
    layer: nn.Linear, weight_init: str = "kaiming_uniform", bias_const: float = 0.0
) -> nn.Linear:
    """Initialize layer weights."""
    if weight_init == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(
            layer.weight, nonlinearity="relu"
        )  # kaiming uniform best for ReLU
    elif weight_init == "uniform":
        torch.nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        # optional uniform init for the last layer (from DDPG paper)
    else:
        raise ValueError(
            "Unknown weight initialization; use 'kaiming_uniform' or 'uniform'"
        )
    torch.nn.init.constant_(layer.bias, bias_const)  # start with 0 bias
    return layer


def mlp(
    sizes: tuple[int],
    activation: nn.Module,
    output_activation: nn.Module = nn.Identity(),
    output_weight_init: str = "kaiming_uniform",
) -> nn.Sequential:
    """Create a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [
                layer_init(nn.Linear(sizes[j], sizes[j + 1])),
                activation,
            ]
        else:
            layers += [
                layer_init(
                    nn.Linear(sizes[j], sizes[j + 1]), weight_init=output_weight_init
                ),
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
        self.mu_layer = layer_init(
            nn.Linear(hidden_sizes[-1], act_dim), weight_init="uniform"
        )
        self.log_std_layer = layer_init(
            nn.Linear(hidden_sizes[-1], act_dim), weight_init="uniform"
        )
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
        self, obs: torch.Tensor, deterministic: bool = False, with_log_prob: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            x_t = mu
        else:
            # rsample keeps computation graph alive using reparameterization trick
            # meanwhile, sample does not
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
        self.net = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            activation,
            output_weight_init="uniform",
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(
            self.net(torch.cat([obs, act], dim=-1)), -1
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
        # Some envs define the action space dims as a double, so we need to force float
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
        # Note we use Q(s,a) as the value function
        self.q1 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            """Sample action from actor."""
            return self.pi.forward(obs, deterministic, False)[0]


# From OpenAI SpinningUp, may be useful later:


def combined_shape(length: int, shape: int | tuple[int] | None = None) -> tuple[int]:
    """Create a tuple of size (length, shape)."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: nn.Module) -> int:
    """Count the number of parameters in a module."""
    return sum([np.prod(p.shape) for p in module.parameters()])
