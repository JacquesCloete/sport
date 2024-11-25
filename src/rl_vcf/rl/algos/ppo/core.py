# Credit to OpenAI Spinning Up

import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def combined_shape(length: int, shape: int | tuple[int] | None=None) -> tuple[int]:
    """Create a tuple of size (length, shape)."""
    if shape is None:
        return(length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes: tuple[int], activation=nn.Module, output_activation=nn.Identity) -> nn.Sequential:
    """Create a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module: nn.Module) -> int:
    """Count the number of parameters in a module."""
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x : np.ndarray, discount : float) -> np.ndarray:
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

class Actor(nn.Module):
    """Actor Base Class."""

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi: torch.distributions.Distribution, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor | None=None) -> tuple[torch.distributions.Distribution, torch.Tensor | None]:
        # Produce action distribution for given observations, and optionally compute the log likelihood of given actions under those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
class MLPCategoricalActor(Actor):
    """Categorical Actor."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int], activation: nn.Module):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi: torch.distributions.Distribution, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act)
    
LOG_STD_MAX = 2
LOG_STD_MIN = -20
    
class MLPGaussianActor(Actor):
    """Gaussian Actor."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int], activation: nn.Module):
        super().__init__()
        log_std = 