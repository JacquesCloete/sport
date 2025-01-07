import math
import pickle
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
        self, num_envs: int, max_episode_length: int, max_num_scenarios: int
    ) -> None:
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.max_num_scenarios = max_num_scenarios
        self.reset_all()

    def reset_all(self) -> None:
        """Reset the scenario database."""
        self.scenario_data = np.full(
            (self.max_num_scenarios, self.max_episode_length + 1), False, dtype=bool
        )
        self.times_taken = np.zeros(self.max_num_scenarios, dtype=int)
        self.active_scenarios = np.arange(self.num_envs, dtype=int)
        self.num_collected_scenarios = 0
        self.timesteps = np.zeros(self.num_envs, dtype=int)
        self.epsilons_dict = {}

    def reset_active_scenarios(self) -> None:
        """Reset the active scenarios."""
        self.timesteps = np.zeros(self.num_envs, dtype=int)
        self.scenario_data[self.active_scenarios] = False

    def update(
        self,
        done: Container[bool],
        goal_achieved: Container[bool],
        constraint_violated: Container[bool],
    ) -> None:
        """Update the scenario database."""
        # For each active scenario, check task status
        if not any(self.active_scenarios < self.max_num_scenarios):
            warning(
                "Max no. scenarios have already been collected; this update will be ignored."
            )
        else:
            for p_idx in range(self.num_envs):
                # If the scenario is to be collected
                if self.active_scenarios[p_idx] < self.max_num_scenarios:
                    # If the task is done:
                    if done[p_idx]:
                        # Record time taken for scenario
                        self.times_taken[self.active_scenarios[p_idx]] = self.timesteps[
                            p_idx
                        ]
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
                        # Increment number of collected scenarios
                        self.num_collected_scenarios += 1
                        # Reset the time step for the next scenario
                        self.timesteps[p_idx] = 0
                    else:
                        # If scenario is ongoing, increment pointer time step
                        self.timesteps[p_idx] += 1

    def get_num_successes(self) -> NDArray:
        """Get the number of successes in the scenario database by each timestep."""
        return self.scenario_data[: self.num_collected_scenarios].sum(axis=0)

    def get_num_failures(self) -> NDArray:
        """Get the number of failures in the scenario database by each timestep."""
        return np.logical_not(self.scenario_data[: self.num_collected_scenarios]).sum(
            axis=0
        )

    def get_failure_rate(self) -> NDArray:
        """Get the failure rate in the scenario database by each timestep."""
        return self.get_num_failures() / self.num_collected_scenarios

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
                        N=self.num_collected_scenarios,
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
        self,
        conf: float,
        force_recompute: bool = True,
        tol: float = 1e-5,
        cutoff: float = 1.0,
        it_max: int = 100,
    ) -> NDArray:
        """
        Get epsilon for each timestep, for a given confidence level.

        Also saves epsilons in a dictionary to avoid recomputation.

        Note that increasing confidence level takes exponentially longer to compute.
        """
        if (conf not in self.epsilons_dict) or force_recompute:
            # Compute epsilons
            unique_ks = np.unique(self.get_num_failures())
            epsilon_dict = dict(
                zip(
                    unique_ks,
                    estimate_epsilon_parallel(
                        conf=conf,
                        N=self.num_collected_scenarios,
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
        force_recompute: bool = True,
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
                epsilons = self.get_num_failures() / self.num_collected_scenarios
                epsilons = np.minimum(epsilons * alpha_array, 1.0)
                ax.plot(epsilons, label=emp_str)
        for conf in confs:
            epsilons = self.get_all_epsilons(
                conf=conf,
                force_recompute=force_recompute,
                tol=tol,
                cutoff=cutoff,
                it_max=it_max,
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
        title_str = (
            f"Policy Failure Probability Bounds, N={self.num_collected_scenarios}"
        )
        ax.title.set_text(title_str)
        return fig, ax

    def get_successful_times_taken(self) -> NDArray:
        """Get the number of timesteps taken for each successful scenario."""
        return self.times_taken[
            np.sum(self.scenario_data[: self.num_collected_scenarios], axis=1) > 0
        ]

    def get_mean_successful_time_taken(self) -> float:
        """Get the mean number of timesteps taken for successful scenarios."""
        return np.mean(self.get_successful_times_taken())

    def get_std_successful_time_taken(self) -> float:
        """Get the standard deviation of the number of timesteps taken for successful scenarios."""
        return np.std(self.get_successful_times_taken())

    def plot_max_log_alpha(
        self,
        bounds: Container[float],
        conf: float = 0.9999999,
        cutoff: float = 1.0,
    ) -> tuple[Figure, Axes, dict[float, tuple[int, float]]]:
        """
        Plot the maximum permitted log-alpha over all possible maximum episode lengths T for each specified failure bound.

        Also returns the locations of the peaks for each bound (i.e. the T/log-alpha pair that permits the largest alpha for the bound).
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        peaks = {}
        for bound in bounds:
            log_alpha_array = np.maximum(
                (
                    np.log(bound)
                    - np.log(self.get_all_epsilons(conf=conf, cutoff=cutoff)[1:])
                )
                / np.arange(1, self.max_episode_length + 1),
                0.0,
            )  # alpha >= 1.0 required
            label = r"$\epsilon_{{T}}\alpha^{{T}}={bound}, \beta={b:.1E}$".format(
                b=1 - conf, bound=bound
            )
            ax.plot(
                np.arange(1, self.max_episode_length + 1),
                log_alpha_array,
                label=label,
            )

            peaks[bound] = (np.argmax(log_alpha_array) + 1, np.max(log_alpha_array))

        ax.set_xlim([1, self.max_episode_length + 1])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Maximum Episode Length T (time steps)")
        ax.set_ylabel(r"Maximum Permitted ln($\alpha$)")
        ax.grid(which="major")
        ax.grid(which="minor", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        return fig, ax, peaks

    def get_optimal_max_episode_length(
        self,
        bound: float,
        conf: float = 0.9999999,
        cutoff: float = 1.0,
    ) -> tuple[int, float]:
        """
        Get the optimal maximum episode length T for a given bound, as well as the maximum permitted alpha for that bound.
        """
        log_alpha_array = np.maximum(
            (
                np.log(bound)
                - np.log(self.get_all_epsilons(conf=conf, cutoff=cutoff)[1:])
            )
            / np.arange(1, self.max_episode_length + 1),
            0.0,
        )  # alpha >= 1.0 required
        T = np.argmax(log_alpha_array) + 1
        log_alpha = np.max(log_alpha_array)
        return T, np.exp(log_alpha)


def save_scenario_database(db: ScenarioDatabase, filename: str) -> None:
    """Save a scenario database to a file."""
    assert ".pkl" in filename, "filename must have .pkl extension."
    with open(filename, "wb") as file:
        pickle.dump(db, file, pickle.HIGHEST_PROTOCOL)


def load_scenario_database(filename: str) -> ScenarioDatabase:
    """Load a scenario database from a file."""
    assert ".pkl" in filename, "filename must have .pkl extension."
    with open(filename, "rb") as file:
        db = pickle.load(file)
    return db


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


def add_plotting_data_to_dicts(
    scenario_db: ScenarioDatabase,
    alpha: float,
    T: int,
    mean_successful_time_taken: dict[float, float],
    std_successful_time_taken: dict[float, float],
    empirical_failure_rate: dict[float, float],
    posterior_bound_failure_rate: dict[float, float],
    conf: float = 0.9999999,
) -> None:
    """
    Add data to dictionaries for plotting.
    """
    all_successful_times_taken = scenario_db.get_successful_times_taken()
    # Remove successful times taken that are greater than T (since these now correspond to failures)
    successful_times_taken = all_successful_times_taken[all_successful_times_taken <= T]
    mean_successful_time_taken[alpha] = np.mean(successful_times_taken)
    std_successful_time_taken[alpha] = np.std(successful_times_taken)
    empirical_failure_rate[alpha] = (
        scenario_db.get_num_failures()[T]
    ) / scenario_db.num_collected_scenarios
    posterior_bound_failure_rate[alpha] = scenario_db.get_selected_epsilons(T, conf)


def remove_plotting_data_from_dicts(
    alphas: Container[float],
    mean_successful_time_taken: dict[float, float],
    std_successful_time_taken: dict[float, float],
    empirical_failure_rate: dict[float, float],
    posterior_bound_failure_rate: dict[float, float],
) -> None:
    """
    Remove data from dictionaries for plotting.
    """
    for alpha in alphas:
        mean_successful_time_taken.pop(alpha, None)
        std_successful_time_taken.pop(alpha, None)
        empirical_failure_rate.pop(alpha, None)
        posterior_bound_failure_rate.pop(alpha, None)


def plot_mean_std_time_taken(
    mean_successful_time_taken: dict[float, float],
    std_successful_time_taken: dict[float, float],
) -> tuple[Figure, Axes]:
    """
    Plot the mean and standard deviation of episode length for successful scenarios across different alphas.
    """
    # Extract alpha values and corresponding means and standard deviations for successful time taken
    alphas = np.array(list(mean_successful_time_taken.keys()))
    means = np.array(list(mean_successful_time_taken.values()))
    stds = np.array(list(std_successful_time_taken.values()))

    # Plot the mean with standard deviations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas, means, color="blue", marker="x")
    ax.fill_between(alphas, means - stds, means + stds, alpha=0.2, color="blue")
    ax.fill_between(
        alphas,
        means - stds,
        means + stds,
        alpha=0.5,
        facecolor="none",
        edgecolor="blue",
    )
    ax.set_xlim([alphas.min(), alphas.max()])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Mean \xB1 STD Episode Length on Success (time steps)")
    ax.set_xscale("log")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    return fig, ax


def plot_failure_probs(
    epsilon: float,
    empirical_failure_rate: dict[float, float],
    posterior_bound_failure_rate: dict[float, float],
    T: int,
    conf: float = 0.9999999,
) -> tuple[Figure, Axes]:
    """
    Plot the empirical failure rate and prior/posterior bound on failure probability across different alphas.
    """
    # Extract alpha values and corresponding means and standard deviations for successful time taken
    alphas = np.array(list(empirical_failure_rate.keys()))
    failure_rates = np.array(list(empirical_failure_rate.values()))
    posterior_bounds = np.array(list(posterior_bound_failure_rate.values()))

    fig, ax = plt.subplots(figsize=(10, 5))

    scen_str = r"Prior Scenario-Based Bound ($\beta={b:.1E}$)".format(b=1 - conf)
    # alpha_range = np.linspace(alphas.min(), alphas.max(), 1000)
    alpha_range = np.logspace(np.log10(alphas.min()), np.log10(alphas.max()), 1000)
    prior_bounds = np.minimum(epsilon * alpha_range**T, 1.0)
    ax.plot(alpha_range, prior_bounds, color="green", label=scen_str)

    emp_str = r"Empirical Failure Rate ($\frac{{k}}{{N}}$)"
    ax.plot(alphas, failure_rates, color="blue", label=emp_str, marker="x")

    scen_str = r"Posterior Scenario-Based Bound ($\beta={b:.1E}$)".format(b=1 - conf)
    ax.plot(alphas, posterior_bounds, color="red", label=scen_str, marker="x")

    ax.set_xlim([alphas.min(), alphas.max()])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Failure Probability")
    ax.set_xscale("log")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    ax.legend()
    return fig, ax


def plot_max_log_alpha(
    scenario_db: ScenarioDatabase,
    bounds: Container[float],
    conf: float = 0.9999999,
    cutoff: float = 1.0,
) -> tuple[Figure, Axes, dict[float, tuple[int, float]]]:
    """
    LEGACY FUNCTION FOR OLD SCENARIO DATABASES -- now a method of ScenarioDatabase

    Plot the maximum permitted log-alpha over all possible maximum episode lengths T for each specified failure bound.

    Also returns the locations of the peaks for each bound (i.e. the T/log-alpha pair that permits the largest alpha for the bound).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    peaks = {}
    for bound in bounds:
        log_alpha_array = np.maximum(
            (
                np.log(bound)
                - np.log(scenario_db.get_all_epsilons(conf=conf, cutoff=cutoff)[1:])
            )
            / np.arange(1, scenario_db.max_episode_length + 1),
            0.0,
        )  # alpha >= 1.0 required
        label = r"$\epsilon_{{T}}\alpha^{{T}}={bound}, \beta={b:.1E}$".format(
            b=1 - conf, bound=bound
        )
        ax.plot(
            np.arange(1, scenario_db.max_episode_length + 1),
            log_alpha_array,
            label=label,
        )

        peaks[bound] = (np.argmax(log_alpha_array) + 1, np.max(log_alpha_array))

    ax.set_xlim([1, scenario_db.max_episode_length + 1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Maximum Episode Length T (time steps)")
    ax.set_ylabel(r"Maximum Permitted ln($\alpha$)")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    return fig, ax, peaks


def get_optimal_max_episode_length(
    scenario_db: ScenarioDatabase,
    bound: float,
    conf: float = 0.9999999,
    cutoff: float = 1.0,
) -> tuple[int, float]:
    """
    LEGACY FUNCTION FOR OLD SCENARIO DATABASES -- now a method of ScenarioDatabase

    Get the optimal maximum episode length T for a given bound, as well as the maximum permitted alpha for that bound.
    """
    log_alpha_array = np.maximum(
        (
            np.log(bound)
            - np.log(scenario_db.get_all_epsilons(conf=conf, cutoff=cutoff)[1:])
        )
        / np.arange(1, scenario_db.max_episode_length + 1),
        0.0,
    )  # alpha >= 1.0 required
    T = np.argmax(log_alpha_array) + 1
    log_alpha = np.max(log_alpha_array)
    return T, np.exp(log_alpha)
