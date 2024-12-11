import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from numpy.typing import NDArray

from rl_vcf.rl.algos.ppo.core import MLPCritic, MLPSquashedGaussianActor

ETA = 1e-8


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

        # Build value function
        # Note we use V(s) as the value function
        self.v = MLPCritic(obs_dim, hidden_sizes_critic, activation_critic)

        # Build policy projector as CVXPY optimization problem
        self.projector = ProjectionProblem(act_dim, alpha)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value from critic."""
        return self.v.forward(obs)

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
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample projected action (and log-prob) from actor and get value from critic."""
        # TODO
        pass
        # return (
        #     act,
        #     logp_a,
        #     entropy,
        #     self.v.forward(obs),
        #     x_t,
        # )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from actor."""
        # TODO
        pass
        # return self.pi_task.forward(obs, None, deterministic, False)[0]

    def set_alpha(self, alpha: float = 1.0) -> None:
        assert alpha >= 1.0, "alpha >= required (alpha: {a})".format(a=alpha)
        self.projector.set_alpha(alpha)
