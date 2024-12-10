import math
from concurrent.futures import ProcessPoolExecutor
from logging import warning
from typing import Container

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


class ScenarioDatabase:
    """A class for storing and updating a scenario database."""

    def __init__(
        self, num_envs: int, max_episode_length: int, num_scenarios: int
    ) -> None:
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.num_scenarios = num_scenarios
        self.reset()

    def reset(self) -> None:
        """Reset the scenario database."""
        self.scenario_data = np.full(
            (self.num_scenarios, self.max_episode_length + 1), False, dtype=bool
        )
        self.active_scenarios = np.arange(self.num_envs, dtype=int)
        self.timesteps = np.zeros(self.num_envs, dtype=int)
        self.epsilons_dict = {}

    def update(
        self,
        done: Container[bool],
        goal_achieved: Container[bool],
        constraint_violated: Container[bool],
    ) -> None:
        """Update the scenario database."""
        # For each active scenario, check task status
        if not any(self.active_scenarios < self.num_scenarios):
            warning(
                "All scenarios have already been collected; this update will be ignored."
            )
        else:
            for p_idx in range(self.num_envs):
                # If the scenario is to be collected
                if self.active_scenarios[p_idx] < self.num_scenarios:
                    # If the task is done:
                    if done[p_idx]:
                        # If the task is done because the goal was achieved, mark the task as successful
                        if goal_achieved[p_idx]:
                            # On scenario task success, mask current and future time steps as task success
                            # (Assumes reach-avoid task with no consideration of failure after success)
                            self.scenario_data[self.active_scenarios[p_idx]][
                                self.timesteps[p_idx] :
                            ] = True
                        elif constraint_violated[p_idx]:
                            # On constraint violation, mark the task as unsuccessful
                            self.scenario_data[self.active_scenarios[p_idx]][
                                self.timesteps[p_idx] :
                            ] = False
                        else:
                            # else, the task was not successful as it ran out of time
                            self.scenario_data[self.active_scenarios[p_idx]][
                                self.timesteps[p_idx] :
                            ] = False
                        # The scenario pointer now tracks next scenario
                        self.active_scenarios[p_idx] = np.max(self.active_scenarios) + 1
                        # Reset the time step for the next scenario
                        self.timesteps[p_idx] = 0
                    else:
                        # If scenario is ongoing, increment pointer time step
                        self.timesteps[p_idx] += 1

    def get_num_successes(self) -> NDArray:
        """Get the number of successes in the scenario database by each timestep."""
        return self.scenario_data.sum(axis=0)

    def get_num_failures(self) -> NDArray:
        """Get the number of failures in the scenario database by each timestep."""
        return np.logical_not(self.scenario_data).sum(axis=0)

    def get_selected_epsilons(
        self,
        t_steps: Container[int],
        conf: float,
        tol: float = 1e-5,
        cutoff: float = 1.0,
        it_max: int = 100,
    ) -> NDArray:
        """
        Get epsilon for specific timesteps, for a given confidence level.

        Does not save epsilons.

        Note that increasing confidence level takes exponentially longer to compute.
        """
        t_steps_array = np.array(t_steps)
        if conf not in self.epsilons_dict:
            # Compute epsilons if not already in dictionary
            unique_ks = np.unique(self.get_num_failures()[t_steps_array])
            epsilon_dict = dict(
                zip(
                    unique_ks,
                    estimate_epsilon_parallel(
                        conf=conf,
                        N=self.num_scenarios,
                        k=unique_ks,
                        tol=tol,
                        cutoff=cutoff,
                        it_max=it_max,
                    ),
                )
            )
            epsilons = np.array(
                [epsilon_dict[k] for k in self.get_num_failures()[t_steps_array]]
            )
        else:
            epsilons = self.epsilons_dict[conf][t_steps_array]
        return epsilons

    def get_all_epsilons(
        self, conf: float, tol: float = 1e-5, cutoff: float = 1.0, it_max: int = 100
    ) -> NDArray:
        """
        Get epsilon for each timestep, for a given confidence level.

        Also saves epsilons in a dictionary to avoid recomputation.

        Note that increasing confidence level takes exponentially longer to compute.
        """
        # TODO: saving does not account for different tol and cutoff values
        if conf not in self.epsilons_dict:
            # Compute epsilons if not already in dictionary
            unique_ks = np.unique(self.get_num_failures())
            epsilon_dict = dict(
                zip(
                    unique_ks,
                    estimate_epsilon_parallel(
                        conf=conf,
                        N=self.num_scenarios,
                        ks=unique_ks,
                        tol=tol,
                        cutoff=cutoff,
                        it_max=it_max,
                    ),
                )
            )
            epsilons = np.array([epsilon_dict[k] for k in self.get_num_failures()])
            # Save to dictionary
            self.epsilons_dict[conf] = epsilons
        return self.epsilons_dict[conf]

    def plot_epsilons(
        self,
        confs: Container[float],
        plot_empirical: bool = False,
        alphas: Container[float] = [1.0],
        tol: float = 1e-5,
        cutoff: float = 1.0,
        it_max: int = 100,
    ) -> tuple[Figure, Axes]:
        """
        Create a plot of epsilon for each timestep.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        indices = np.arange(self.max_episode_length + 1)

        if plot_empirical:
            for alpha in alphas:
                alpha_array = alpha**indices
                emp_str = r"$\alpha={a}, \frac{{k}}{{N}}$ (empirical)".format(
                    a=alpha,
                )
                epsilons = self.get_num_failures() / self.num_scenarios
                epsilons = np.minimum(epsilons * alpha_array, 1.0)
                ax.plot(epsilons, label=emp_str)
        for conf in confs:
            epsilons = self.get_all_epsilons(
                conf=conf, tol=tol, cutoff=cutoff, it_max=it_max
            )
            for alpha in alphas:
                alpha_array = alpha**indices
                scen_str = r"$\alpha={a}, \beta={b:.1E}$".format(b=1 - conf, a=alpha)
                ax.plot(np.minimum(epsilons * alpha_array, 1.0), label=scen_str)
        ax.set_xlim([0, self.max_episode_length])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Time step, t")
        ax.set_ylabel("Bound on failure probability, $\epsilon_{t}$")
        ax.minorticks_on()
        ax.grid(which="major")
        ax.grid(which="minor", linestyle="--", alpha=0.5)
        ax.legend()
        title_str = f"Policy Failure Probability Bounds, N={self.num_scenarios}"
        ax.title.set_text(title_str)
        return fig, ax


def log_binomial_coefficient(N: int, k: int) -> float:
    """
    Compute the logarithm of the binomial coefficient "N choose k".
    """
    if k < 0 or k > N:
        return float("-inf")
    if k == 0 or k == N:
        return 0.0
    return sum(math.log(i) for i in range(N, N - k, -1)) - sum(
        math.log(i) for i in range(1, k + 1)
    )


def estimate_epsilon(
    conf: float,
    N: int,
    k: int,
    tol: float = 0.0,
    cutoff: float = 1.0,
    it_max: int = 100,
) -> float:
    """
    Numerically estimate epsilon for a given confidence level.

    Note that the function returns an upper bound on epsilon.
    """
    # TODO: The binomial coefficient explodes when N is large (somewhere 10000 > N > 1000)
    # as k increases, making numerical estimation impossible.
    # For cases where N is large and k is not small, we will need to settle for less tight bound
    # that is still computationally feasible.

    # Compute beta
    beta = 1.0 - conf

    if k == 0:
        # If there are no violations, we can directly compute epsilon
        return 1.0 - beta ** (1 / N)

    elif k == N:
        # If all scenarios have violations, epsilon is trivially 1
        return 1.0

    elif N <= 1000 and k / N <= cutoff:
        # Numerically compute a tight bound for epsilon, if computationally feasible
        # Uses a divide-and-conquer style approach

        # Initialize epsilon bounds
        eps_lower = 0.0
        eps_upper = 1.0

        # Precompute coefficients
        binom_coeffs = np.array([math.comb(N, i) for i in range(k + 1)])

        # Precompute indices
        indices = np.arange(k + 1)

        for _ in range(it_max):

            # Trial epsilon is midway between current bounds
            eps_trial = (eps_lower + eps_upper) / 2

            # Compute beta for trial epsilon using vectorized operations
            eps_powers = eps_trial**indices
            one_minus_eps_powers = (1 - eps_trial) ** (N - indices)
            beta_trial = np.sum(binom_coeffs * eps_powers * one_minus_eps_powers)

            # If trial beta is lower than target beta, we set the upper bound to the trial epsilon
            if beta_trial <= beta:
                eps_upper = eps_trial
            # Otherwise, we set the lower bound to the trial epsilon
            else:
                eps_lower = eps_trial

            # Early stopping condition
            if eps_upper - eps_lower <= tol:
                break

        return eps_upper

    elif k / N <= cutoff:
        # Numerically compute a tight bound for epsilon, if computationally feasible
        # Uses a divide-and-conquer style approach
        # Puts things in log form for numerical stability

        # Initialize epsilon bounds
        eps_lower = 0.0
        eps_upper = 1.0

        # Precompute coefficients
        log_binom_coeffs = np.array(
            [log_binomial_coefficient(N, i) for i in range(k + 1)], dtype=float
        )
        log_eps_coeffs = np.arange(start=0, stop=k + 1, step=1, dtype=float)
        log_one_minus_eps_coeffs = np.arange(
            start=N, stop=N - k - 1, step=-1, dtype=float
        )

        for _ in range(it_max):

            # Trial epsilon is midway between current bounds
            eps_trial = (eps_lower + eps_upper) / 2

            # Compute beta for trial epsilon using vectorized operations
            log_beta_terms = (
                log_binom_coeffs
                + math.log(eps_trial) * log_eps_coeffs
                + math.log(1 - eps_trial) * log_one_minus_eps_coeffs
            )
            beta_trial = np.sum(np.exp(log_beta_terms))

            # If trial beta is lower than target beta, we set the upper bound to the trial epsilon
            if beta_trial <= beta:
                eps_upper = eps_trial
            # Otherwise, we set the lower bound to the trial epsilon
            else:
                eps_lower = eps_trial

            # Early stopping condition
            if eps_upper - eps_lower <= tol:
                break

        return eps_upper

    else:
        # If numerical computation of a tight bound is not needed,
        # return a loose bound on epsilon (Chernoff bound for Binomial tail)

        return (
            1
            / N
            * (
                (k - math.log(beta))
                + math.sqrt(math.log(beta) * (math.log(beta) - 2 * k))
            )
        )


def estimate_epsilon_parallel(
    conf: float,
    N: int,
    ks: Container[int],
    tol: float = 0.0,
    cutoff: float = 1.0,
    it_max: int = 100,
) -> list[float]:
    """
    Parallelized epsilon estimation across different values of k using multiprocessing.
    """
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                estimate_epsilon,
                conf=conf,
                N=N,
                k=k,
                tol=tol,
                cutoff=cutoff,
                it_max=it_max,
            )
            for k in ks
        ]
        results = [future.result() for future in futures]
    return results
