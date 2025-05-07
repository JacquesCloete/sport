import warnings

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from numpy.typing import NDArray
from torch.distributions import Normal

from sport.rl.algos.ppo.core import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    MLPCritic,
    MLPSquashedGaussianActor,
)

ETA = 1e-8
TOL = 1e-7


def max_log_diag_gaussian_ratio(
    mu1: NDArray, std1: NDArray, mu2: NDArray, std2: NDArray
) -> NDArray:
    """
    Compute the maximum log probability ratio between two diagonal Gaussian distributions.

    max_log_gaussian_ratio(mu1, std1, mu2, std2) = log(max_x N(mu2, std2)(x) / N(mu1, std1)(x))

    The log is for numerical stability once the ratio gets large.
    """
    if np.any(std2 > std1):
        # If the second distribution has a larger std, the ratio is infinite
        return np.inf
    elif (
        np.all(mu1 - TOL <= mu2)
        and np.all(mu2 <= mu1 + TOL)
        and np.all(std1 - TOL <= std2)
    ):
        # If the two distributions are the same, the ratio is 1 (so log ratio is 0)
        return 0.0
    elif np.any(std2 == std1):
        # If the second distribution has an equal std, the ratio (and log ratio) is infinite
        return np.inf
    else:
        return (
            np.log(std1).sum()
            - np.log(std2).sum()
            + 0.5 * ((mu2 - mu1) ** 2 / (std1**2 - std2**2)).sum()
        )


class ProjectionProblem:
    """CVXPY optimization problem for policy projection."""

    def __init__(self, act_dim: int, alpha: float = 1.0) -> None:
        # Initialize variables and parameters
        # To avoid confusion, prefix all CVXPY objects with "cp_"

        # Projection ratio parameter
        assert alpha >= 1.0, "alpha >= required (alpha: {a})".format(a=alpha)
        self.cp_alpha = cp.Parameter(value=alpha, nonneg=True)

        # Base policy parameters
        self.cp_mu_base = cp.Parameter(act_dim)
        self.cp_sig_base = cp.Parameter(act_dim, pos=True)

        # Task policy parameters
        self.cp_mu_task = cp.Parameter(act_dim)
        self.cp_sig_task = cp.Parameter(act_dim, pos=True)

        # Projected policy variables
        self.cp_mu_proj = cp.Variable(act_dim)
        self.cp_sig_proj = cp.Variable(act_dim, pos=True)

        # Additional variables/parameters for DPP:
        # Base policy
        self.cp_sig_base_inv = cp.Parameter(act_dim, pos=True)  # 1/sig_base
        self.cp_sig_sig_base = cp.Parameter(act_dim, pos=True)  # sig_base^2

        # Task policy
        self.cp_sig_task_inv = cp.Parameter(act_dim, pos=True)  # 1/sig_task
        self.cp_mu_sig_task = cp.Parameter(act_dim)  # mu_task/sig_task
        self.cp_mu_sig_sig_task = cp.Parameter(act_dim)  # mu_task/(sig_task)^2

        # Sum term
        self.cp_mu_base_var = cp.Variable(act_dim)  # mu_base
        self.cp_sig_sig_base_var = cp.Variable(act_dim, pos=True)  # sig_base^2

        # Problem definition (as DPP)
        cp_objective = cp.Minimize(
            -2 * cp.sum(cp.log(self.cp_sig_proj))
            + cp.sum_squares(cp.multiply(self.cp_sig_proj, self.cp_sig_task_inv))
            + cp.sum_squares(cp.multiply(self.cp_mu_proj, self.cp_sig_task_inv))
            - 2 * cp.sum(cp.multiply(self.cp_mu_proj, self.cp_mu_sig_sig_task))
            + cp.sum_squares(self.cp_mu_sig_task)
        )

        cp_constraints = []

        cp_constraint_1 = self.cp_sig_proj + ETA <= self.cp_sig_base
        cp_constraints.append(cp_constraint_1)

        # we define the sum term like this so that we can use quad_over_lin and thus allow the
        # problem to be DCP (standard quotient operator is not DCP)
        # note that the numerator gets squared (which we want anyway)
        cp_sum_term = 0
        for i in range(0, act_dim):
            cp_sum_term += cp.quad_over_lin(
                self.cp_mu_proj[i] - self.cp_mu_base_var[i],
                self.cp_sig_sig_base_var[i] - cp.square(self.cp_sig_proj[i]),
            )

        cp_constraint_2 = -cp.sum(cp.log(self.cp_sig_base_inv)) - cp.sum(
            cp.log(self.cp_sig_proj)
        ) + (1 / 2) * cp_sum_term <= cp.log(self.cp_alpha)
        cp_constraints.append(cp_constraint_2)

        cp_constraint_3 = self.cp_mu_base_var == self.cp_mu_base
        cp_constraints.append(cp_constraint_3)

        cp_constraint_4 = self.cp_sig_sig_base_var == self.cp_sig_sig_base
        cp_constraints.append(cp_constraint_4)

        self.cp_problem = cp.Problem(cp_objective, cp_constraints)

    def solve(
        self,
        mu_base: NDArray,
        std_base: NDArray,
        mu_task: NDArray,
        std_task: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Project the task policy distribution onto the feasible set around the base policy distribution."""
        # If policy ratio is set to 1, just return the base policy distribution
        if self.cp_alpha.value <= 1.0:
            return mu_base, std_base

        else:
            # Set projection problem parameters
            self.cp_mu_base.value = mu_base
            self.cp_sig_base.value = std_base

            self.cp_mu_task.value = mu_task
            self.cp_sig_task.value = std_task

            self.cp_sig_base_inv.value = 1.0 / std_base
            self.cp_sig_sig_base.value = np.square(std_base)

            self.cp_sig_task_inv.value = 1 / std_task
            self.cp_mu_sig_task.value = mu_task / std_task
            self.cp_mu_sig_sig_task.value = mu_task / np.square(std_task)

            # Project onto feasible set
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = self.cp_problem.solve()

            # Extract projected mean and std
            return self.cp_mu_proj.value, self.cp_sig_proj.value

    def set_alpha(self, alpha: float = 1.0) -> None:
        assert alpha >= 1.0, "alpha >= required (alpha: {a})".format(a=alpha)
        self.cp_alpha.value = alpha


class MLPProjectedActorCritic(nn.Module):
    """Projected actor and critic."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_sizes_base: tuple[int] = (64, 64),
        activation_base: nn.Module = nn.Tanh(),
        hidden_sizes_task: tuple[int] = (64, 64),
        activation_task: nn.Module = nn.Tanh(),
        hidden_sizes_critic: tuple[int] = (64, 64),
        activation_critic: nn.Module = nn.Tanh(),
        alpha: float = 1.0,
        check_max_policy_ratios: bool = False,
        check_ref_task_policy: bool = False,
        hidden_sizes_ref_task: tuple[int] = (64, 64),
        activation_ref_task: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        obs_dim = np.array(observation_space.shape).prod()

        # Policy builder depends on action space
        act_dim = np.prod(action_space.shape)
        # Some envs define the action space dims as a double, so we need to force float
        act_scale = torch.tensor(
            (action_space.high - action_space.low) / 2.0, dtype=torch.float32
        )
        act_bias = torch.tensor(
            (action_space.high + action_space.low) / 2.0, dtype=torch.float32
        )

        # Base policy
        self.pi_base = MLPSquashedGaussianActor(
            obs_dim,
            act_dim,
            hidden_sizes_base,
            activation_base,
            act_scale,
            act_bias,
        )

        # Freeze base policy parameters
        self.pi_base.requires_grad_(False)

        # Task policy
        self.pi_task = MLPSquashedGaussianActor(
            obs_dim,
            act_dim,
            hidden_sizes_task,
            activation_task,
            act_scale,
            act_bias,
        )

        self.check_max_policy_ratios = check_max_policy_ratios
        self.check_ref_task_policy = check_ref_task_policy

        if self.check_ref_task_policy:

            # reference task policy (used to compare against current task policy)
            self.pi_ref_task = MLPSquashedGaussianActor(
                obs_dim,
                act_dim,
                hidden_sizes_ref_task,
                activation_ref_task,
                act_scale,
                act_bias,
            )

            # Freeze reference task policy parameters
            self.pi_ref_task.requires_grad_(False)

        # Build value function
        # Note we use V(s) as the value function
        self.v = MLPCritic(obs_dim, hidden_sizes_critic, activation_critic)

        # Build policy projector as CVXPY optimization problem
        self.projector = ProjectionProblem(act_dim, alpha)

        # Store these for plotting/debugging
        self.latest_mu_base = None
        self.latest_std_base = None
        self.latest_mu_task = None
        self.latest_std_task = None
        self.latest_mu_proj = None
        self.latest_std_proj = None
        self.latest_mu_ref_task = None
        self.latest_std_ref_task = None
        self.latest_mu_ref_proj = None
        self.latest_std_ref_proj = None
        self.latest_log_task_base = None
        self.latest_log_proj_base = None
        self.latest_log_ref_task_base = None
        self.latest_log_ref_proj_base = None

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value from critic."""
        return self.v.forward(obs)

    def get_base_action_and_value(
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (and log-prob) from base policy actor and get value from critic."""
        act, logp_a, entropy, x_t = self.pi_base.forward(
            obs, act, deterministic=False, with_log_prob=True
        )
        return (
            act,
            logp_a,
            entropy,
            self.v.forward(obs),
            x_t,
        )

    def get_unprojected_action_and_value(
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample unprojected action (and log-prob) from actor and get value from critic."""
        act, logp_a, entropy, x_t = self.pi_task.forward(
            obs, act, deterministic=False, with_log_prob=True
        )
        return (
            act,
            logp_a,
            entropy,
            self.v.forward(obs),
            x_t,
        )

    def get_projected_action_and_value(
        self,
        obs: torch.Tensor,
        act: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample projected action (and log-prob) from actor and get value from critic.

        Note that the projection means that we cannot get gradients for the projected action.
        """
        act, logp_a, entropy, x_t = self.pi_task_projected_forward(
            obs,
            act,
            deterministic=False,
            with_log_prob=True,
        )
        return (
            act,
            logp_a,
            entropy,
            self.v.forward(obs),
            x_t,
        )

    def pi_task_projected_forward(
        self,
        obs: torch.Tensor,
        pi_action: torch.Tensor | None = None,  # this must be untransformed action x_t
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:

        # Get base policy distribution
        net_out_base = self.pi_base.net(obs)
        mu_base = self.pi_base.mu_layer(net_out_base)
        log_std_base = self.pi_base.log_std_layer(net_out_base)
        log_std_base = torch.tanh(log_std_base)
        log_std_base = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std_base + 1
        )
        std_base = torch.exp(log_std_base)

        # Get task policy distribution
        net_out_task = self.pi_task.net(obs)
        mu_task = self.pi_task.mu_layer(net_out_task)
        log_std_task = self.pi_task.log_std_layer(net_out_task)
        log_std_task = torch.tanh(log_std_task)
        log_std_task = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std_task + 1
        )
        std_task = torch.exp(log_std_task)

        # Convert to numpy arrays
        np_mu_base = mu_base.detach().cpu().numpy()
        np_std_base = std_base.detach().cpu().numpy()
        np_mu_task = mu_task.detach().cpu().numpy()
        np_std_task = std_task.detach().cpu().numpy()

        np_mu_proj = np.zeros_like(np_mu_task)
        np_std_proj = np.zeros_like(np_std_task)

        # Project task policy onto feasible set around base policy
        if np.size(mu_task.shape) == 1:
            # Single action
            np_mu_proj, np_std_proj = self.projector.solve(
                np_mu_base, np_std_base, np_mu_task, np_std_task
            )
        else:
            # Batch of actions
            for i in range(np_mu_task.shape[0]):
                np_mu_proj[i], np_std_proj[i] = self.projector.solve(
                    np_mu_base[i], np_std_base[i], np_mu_task[i], np_std_task[i]
                )

        self.latest_mu_base = np_mu_base
        self.latest_std_base = np_std_base
        self.latest_mu_task = np_mu_task
        self.latest_std_task = np_std_task
        self.latest_mu_proj = np_mu_proj
        self.latest_std_proj = np_std_proj

        if self.check_ref_task_policy:
            # Get reference task policy distribution
            net_out_ref_task = self.pi_ref_task.net(obs)
            mu_ref_task = self.pi_ref_task.mu_layer(net_out_ref_task)
            log_std_ref_task = self.pi_ref_task.log_std_layer(net_out_ref_task)
            log_std_ref_task = torch.tanh(log_std_ref_task)
            log_std_ref_task = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                log_std_ref_task + 1
            )
            std_ref_task = torch.exp(log_std_ref_task)

            # Convert to numpy arrays
            np_mu_ref_task = mu_ref_task.detach().cpu().numpy()
            np_std_ref_task = std_ref_task.detach().cpu().numpy()

            # Project reference task policy onto feasible set around base policy
            np_mu_ref_proj = np.zeros_like(np_mu_ref_task)
            np_std_ref_proj = np.zeros_like(np_std_ref_task)

            if np.size(mu_task.shape) == 1:
                # Single action
                np_mu_ref_proj, np_std_ref_proj = self.projector.solve(
                    np_mu_base, np_std_base, np_mu_ref_task, np_std_ref_task
                )
            else:
                # Batch of actions
                for i in range(np_mu_ref_task.shape[0]):
                    np_mu_ref_proj[i], np_std_ref_proj[i] = self.projector.solve(
                        np_mu_base[i],
                        np_std_base[i],
                        np_mu_ref_task[i],
                        np_std_ref_task[i],
                    )
        else:
            np_mu_ref_task = None
            np_std_ref_task = None
            np_mu_ref_proj = None
            np_std_ref_proj = None

        self.latest_mu_ref_task = np_mu_ref_task
        self.latest_std_ref_task = np_std_ref_task
        self.latest_mu_ref_proj = np_mu_ref_proj
        self.latest_std_ref_proj = np_std_ref_proj

        if self.check_max_policy_ratios:
            # Compute max policy ratios
            if np.size(mu_task.shape) == 1:
                log_task_base = max_log_diag_gaussian_ratio(
                    np_mu_base, np_std_base, np_mu_task, np_std_task
                )
                log_proj_base = max_log_diag_gaussian_ratio(
                    np_mu_base, np_std_base, np_mu_proj, np_std_proj
                )
            else:
                log_task_base = np.zeros(np_mu_task.shape[0])
                log_proj_base = np.zeros(np_mu_task.shape[0])
                for i in range(np_mu_task.shape[0]):
                    log_task_base[i] = max_log_diag_gaussian_ratio(
                        np_mu_base[i], np_std_base[i], np_mu_task[i], np_std_task[i]
                    )
                    log_proj_base[i] = max_log_diag_gaussian_ratio(
                        np_mu_base[i], np_std_base[i], np_mu_proj[i], np_std_proj[i]
                    )
            if self.check_ref_task_policy:
                if np.size(mu_task.shape) == 1:
                    log_ref_task_base = max_log_diag_gaussian_ratio(
                        np_mu_base, np_std_base, np_mu_ref_task, np_std_ref_task
                    )
                    log_ref_proj_base = max_log_diag_gaussian_ratio(
                        np_mu_base, np_std_base, np_mu_ref_proj, np_std_ref_proj
                    )
                else:
                    log_ref_task_base = np.zeros(np_mu_task.shape[0])
                    log_ref_proj_base = np.zeros(np_mu_task.shape[0])
                    for i in range(np_mu_task.shape[0]):
                        log_ref_task_base[i] = max_log_diag_gaussian_ratio(
                            np_mu_base[i],
                            np_std_base[i],
                            np_mu_ref_task[i],
                            np_std_ref_task[i],
                        )
                        log_ref_proj_base[i] = max_log_diag_gaussian_ratio(
                            np_mu_base[i],
                            np_std_base[i],
                            np_mu_ref_proj[i],
                            np_std_ref_proj[i],
                        )
            else:
                log_ref_task_base = None
                log_ref_proj_base = None
        else:
            log_task_base = None
            log_proj_base = None
            log_ref_task_base = None
            log_ref_proj_base = None

        self.latest_log_task_base = log_task_base
        self.latest_log_proj_base = log_proj_base
        self.latest_log_ref_task_base = log_ref_task_base
        self.latest_log_ref_proj_base = log_ref_proj_base

        # Convert back to torch tensors
        mu = torch.Tensor(np_mu_proj).to(mu_task.device)
        std = torch.Tensor(np_std_proj).to(std_task.device)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        x_t = pi_action
        if x_t is None:
            if deterministic:
                x_t = mu
            else:
                x_t = pi_distribution.rsample()
        y_t = torch.tanh(x_t)
        pi_action = y_t * self.pi_task.act_scale + self.pi_task.act_bias

        if with_log_prob:
            logp_pi = pi_distribution.log_prob(x_t)
            # Enforcing action bounds
            logp_pi -= torch.log(self.pi_task.act_scale * (1 - y_t.pow(2)) + 1e-6)
            logp_pi = logp_pi.sum(1, keepdim=True).squeeze(-1)
            # why doesn't cleanrl squeeze?
        else:
            logp_pi = None

        return (
            pi_action,
            logp_pi,
            pi_distribution.entropy(),
            x_t,
        )

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample projected action from actor."""
        action, _, _, _ = self.pi_task_projected_forward(
            obs=obs,
            pi_action=None,
            deterministic=deterministic,
            with_log_prob=False,
        )
        return action

    def set_alpha(self, alpha: float = 1.0) -> None:
        assert alpha >= 1.0, "alpha >= required (alpha: {a})".format(a=alpha)
        self.projector.set_alpha(alpha)
