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

from sport.rl.algos.projected_ppo.core import MLPProjectedActorCritic
from sport.rl.utils import kl_divergence_diag_gaussians

ETA = 1e-6


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
                        ks=unique_ks,
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
        fig_size_scale_factor: float = 1.0,
    ) -> tuple[Figure, Axes]:
        """
        Create a plot of epsilon for each timestep.
        """
        fig, ax = plt.subplots(
            figsize=(10 * fig_size_scale_factor, 5 * fig_size_scale_factor)
        )

        indices = np.arange(self.max_episode_length + 1)

        if plot_empirical:
            for alpha in reversed(alphas):
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
            for alpha in reversed(alphas):
                alpha_array = alpha**indices
                if len(confs) > 1:
                    scen_str = r"$\alpha={a:}, \beta={b:.1E}$".format(
                        b=1 - conf, a=alpha
                    )
                else:
                    scen_str = r"$\alpha={a:}$".format(a=alpha)
                ax.plot(np.minimum(epsilons * alpha_array, 1.0), label=scen_str)
        ax.set_xlim([0, self.max_episode_length])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Maximum Episode Length, T (time steps)")
        ax.set_ylabel(
            r"Prior Bound on Violation Probability, $\epsilon_{task} = \epsilon_{base}(T) \alpha^{T}$"
        )
        ax.minorticks_on()
        ax.grid(which="major")
        ax.grid(which="minor", linestyle="--", alpha=0.5)
        ax.legend()
        if len(confs) > 1:
            title_str = r"N={scenarios} Scenarios".format(
                scenarios=self.num_collected_scenarios
            )
        else:
            title_str = r"N={scenarios} Scenarios, $\beta={b:.1E}$".format(
                b=1 - conf, scenarios=self.num_collected_scenarios
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
        fig_size_scale_factor: float = 1.0,
    ) -> tuple[Figure, Axes, dict[float, tuple[int, float]]]:
        """
        Plot the maximum permitted log-alpha over all possible maximum episode lengths T for each specified failure bound.

        Also returns the locations of the peaks for each bound (i.e. the T/log-alpha pair that permits the largest alpha for the bound).
        """
        fig, ax = plt.subplots(
            figsize=(10 * fig_size_scale_factor, 5 * fig_size_scale_factor)
        )
        peaks = {}
        for bound in reversed(bounds):
            log_alpha_array = np.maximum(
                (
                    np.log(bound)
                    - np.log(self.get_all_epsilons(conf=conf, cutoff=cutoff)[1:])
                )
                / np.arange(1, self.max_episode_length + 1),
                0.0,
            )  # alpha >= 1.0 required
            label = r"$\epsilon_{{base}}(T)\alpha^{{T}}={bound}".format(bound=bound)
            ax.plot(
                np.arange(1, self.max_episode_length + 1),
                log_alpha_array,
                label=label,
            )

            peaks[bound] = (np.argmax(log_alpha_array) + 1, np.max(log_alpha_array))

        ax.set_xlim([1, self.max_episode_length + 1])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Maximum Episode Length, T (time steps)")
        ax.set_ylabel(r"Maximum Permitted log($\alpha$)")
        ax.minorticks_on()
        ax.grid(which="major")
        ax.grid(which="minor", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.set_title(r"$\beta={b:.1E}$".format(b=1 - conf))

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


class PolicyProjectionDatabase:
    """A class for storing policy projection data."""

    def __init__(self, alpha: float, check_ref_task_policy: bool = False) -> None:

        self.alpha = alpha
        self.check_ref_task_policy = check_ref_task_policy

        self.mu_base = []
        self.std_base = []
        self.mu_task = []
        self.std_task = []
        self.mu_proj = []
        self.std_proj = []

        if self.check_ref_task_policy:
            self.mu_ref_task = []
            self.std_ref_task = []
            self.mu_ref_proj = []
            self.std_ref_proj = []
            self.kl_div_ref_task = []
            self.kl_div_ref_proj = []

        self.episode_count = []

        self.num_collected_data = 0

    def update(self, agent: MLPProjectedActorCritic, episode_count: NDArray) -> None:
        """Update the policy projection database."""
        self.mu_base.append(agent.latest_mu_base)
        self.std_base.append(agent.latest_std_base)
        self.mu_task.append(agent.latest_mu_task)
        self.std_task.append(agent.latest_std_task)
        self.mu_proj.append(agent.latest_mu_proj)
        self.std_proj.append(agent.latest_std_proj)

        if self.check_ref_task_policy:
            self.mu_ref_task.append(agent.latest_mu_ref_task)
            self.std_ref_task.append(agent.latest_std_ref_task)
            self.mu_ref_proj.append(agent.latest_mu_ref_proj)
            self.std_ref_proj.append(agent.latest_std_ref_proj)
            self.kl_div_ref_task.append(
                kl_divergence_diag_gaussians(
                    agent.latest_mu_ref_task,
                    agent.latest_std_ref_task,
                    agent.latest_mu_task,
                    agent.latest_std_task,
                )
            )
            self.kl_div_ref_proj.append(
                kl_divergence_diag_gaussians(
                    agent.latest_mu_ref_proj,
                    agent.latest_std_ref_proj,
                    agent.latest_mu_proj,
                    agent.latest_std_proj,
                )
            )

        self.episode_count.append(episode_count.copy())

        self.num_collected_data += 1

    def concatenate_lists(self) -> None:
        """
        Concatenate the lists in the policy projection database into NumPy arrays, overriding them.

        Only do this once you are done updating the database.
        """
        self.mu_base = np.concatenate(self.mu_base)
        self.std_base = np.concatenate(self.std_base)
        self.mu_task = np.concatenate(self.mu_task)
        self.std_task = np.concatenate(self.std_task)
        self.mu_proj = np.concatenate(self.mu_proj)
        self.std_proj = np.concatenate(self.std_proj)

        if self.check_ref_task_policy:
            self.mu_ref_task = np.concatenate(self.mu_ref_task)
            self.std_ref_task = np.concatenate(self.std_ref_task)
            self.mu_ref_proj = np.concatenate(self.mu_ref_proj)
            self.std_ref_proj = np.concatenate(self.std_ref_proj)
            self.kl_div_ref_task = np.concatenate(self.kl_div_ref_task)
            self.kl_div_ref_proj = np.concatenate(self.kl_div_ref_proj)

        self.episode_count = np.concatenate(self.episode_count)

    def plot_policy_projection(
        self,
        idxs: Container[int],
        plot_std: bool = False,
        plot_ref_task: bool = False,
        plot_cmap: bool = False,
        Nx: int = 50,
        Ny: int = 50,
        border: float = 0.01,
        marker_size: int = 10,
        marker_edge_width: int = 2,
        extra_alphas: list[float] = [],
        fig_size_scale_factor: float = 7.0,
    ) -> tuple[Figure, Axes]:
        """
        Plot policy projection data for the selected samples.

        Note that the contour and color map for the mean and std plot assume the std and mean are kept fixed.
        """
        assert self.num_collected_data > 0, "No data has been collected yet."
        assert not isinstance(self.mu_base, list), "Data has not been concatenated yet."
        assert (
            self.mu_base.shape[1] == 2
        ), "Only 2D policy projection plotting is supported."
        if plot_ref_task:
            assert self.check_ref_task_policy, "Reference task data not available."

        N = len(idxs)
        if plot_std:
            fig, axs = plt.subplots(
                N, 2, figsize=(fig_size_scale_factor * 2, fig_size_scale_factor * N)
            )
            axs = axs.ravel()

            for k, idx in enumerate(idxs):

                mu_x_min = min(
                    self.mu_base[idx][0], self.mu_task[idx][0], self.mu_proj[idx][0]
                )
                mu_x_max = max(
                    self.mu_base[idx][0], self.mu_task[idx][0], self.mu_proj[idx][0]
                )

                mu_y_min = min(
                    self.mu_base[idx][1], self.mu_task[idx][1], self.mu_proj[idx][1]
                )
                mu_y_max = max(
                    self.mu_base[idx][1], self.mu_task[idx][1], self.mu_proj[idx][1]
                )

                if plot_ref_task:
                    mu_x_min = min(
                        mu_x_min,
                        self.mu_ref_task[idx][0],
                        self.mu_ref_proj[idx][0],
                    )
                    mu_x_max = max(
                        mu_x_max,
                        self.mu_ref_task[idx][0],
                        self.mu_ref_proj[idx][0],
                    )
                    mu_y_min = min(
                        mu_y_min,
                        self.mu_ref_task[idx][1],
                        self.mu_ref_proj[idx][1],
                    )
                    mu_y_max = max(
                        mu_y_max,
                        self.mu_ref_task[idx][1],
                        self.mu_ref_proj[idx][1],
                    )

                mu_x_min -= border
                mu_x_max += border
                mu_y_min -= border
                mu_y_max += border

                mu_x = np.linspace(mu_x_min, mu_x_max, Nx)
                mu_y = np.linspace(mu_y_min, mu_y_max, Ny)

                policy_ratio = np.zeros((Ny, Nx))
                if plot_ref_task:
                    policy_ratio_ref = np.zeros((Ny, Nx))
                mu = np.zeros(2)

                for i in range(0, Nx):
                    for j in range(0, Ny):
                        mu[0] = mu_x[i]
                        mu[1] = mu_y[j]
                        policy_ratio[j][i] = np.prod(
                            np.multiply(
                                self.std_base[idx] / self.std_proj[idx],
                                np.exp(
                                    1
                                    / 2
                                    * np.square(mu - self.mu_base[idx])
                                    / (
                                        np.square(self.std_base[idx])
                                        - np.square(self.std_proj[idx])
                                    )
                                ),
                            )
                        )
                if plot_ref_task:
                    for i in range(0, Nx):
                        for j in range(0, Ny):
                            mu[0] = mu_x[i]
                            mu[1] = mu_y[j]
                            policy_ratio_ref[j][i] = np.prod(
                                np.multiply(
                                    self.std_base[idx] / self.std_ref_proj[idx],
                                    np.exp(
                                        1
                                        / 2
                                        * np.square(mu - self.mu_base[idx])
                                        / (
                                            np.square(self.std_base[idx])
                                            - np.square(self.std_ref_proj[idx])
                                        )
                                    ),
                                )
                            )
                if plot_cmap:
                    im = axs[2 * k].pcolormesh(mu_x, mu_y, policy_ratio)
                    # fig.colorbar(im, label="Policy Ratio")
                cs = axs[2 * k].contour(
                    mu_x,
                    mu_y,
                    policy_ratio,
                    levels=extra_alphas + [self.alpha],
                    colors="k",
                )
                h, l = cs.legend_elements()
                axs[2 * k].clabel(cs, inline=True, fontsize=10)
                if plot_ref_task:
                    cs_ref = axs[2 * k].contour(
                        mu_x,
                        mu_y,
                        policy_ratio_ref,
                        levels=extra_alphas + [self.alpha],
                        colors="k",
                        linestyles="--",
                    )
                    h_ref, l_ref = cs_ref.legend_elements()
                    axs[2 * k].clabel(cs_ref, inline=True, fontsize=10)
                base_marker = axs[2 * k].plot(
                    self.mu_base[idx][0],
                    self.mu_base[idx][1],
                    "rx",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="base",
                )
                task_marker = axs[2 * k].plot(
                    self.mu_task[idx][0],
                    self.mu_task[idx][1],
                    "bx",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="task",
                )
                proj_marker = axs[2 * k].plot(
                    self.mu_proj[idx][0],
                    self.mu_proj[idx][1],
                    "m+",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="proj",
                )
                if plot_ref_task:
                    task_ref_marker = axs[2 * k].plot(
                        self.mu_ref_task[idx][0],
                        self.mu_ref_task[idx][1],
                        "gx",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label="task (ref)",
                    )
                    proj_ref_marker = axs[2 * k].plot(
                        self.mu_ref_proj[idx][0],
                        self.mu_ref_proj[idx][1],
                        "y+",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label="proj (ref)",
                    )
                axs[2 * k].set_xlabel(r"$\mu_{0}$ (Mean Forward Drive Force)")
                axs[2 * k].set_ylabel(r"$\mu_{1}$ (Mean Turning Velocity)")

                if k == 0:
                    axs[2 * k].set_title("Policy Means")
                    if plot_ref_task:
                        axs[2 * k].legend(
                            [
                                base_marker[0],
                                task_marker[0],
                                proj_marker[0],
                                task_ref_marker[0],
                                proj_ref_marker[0],
                                h[0],
                                h_ref[0],
                            ],
                            [
                                "base",
                                "task",
                                "proj",
                                "task (ref)",
                                "proj (ref)",
                                r"policy ratio = $\alpha$",
                                r"policy ratio = $\alpha$ (ref)",
                            ],
                        )
                    else:
                        axs[2 * k].legend(
                            [
                                base_marker[0],
                                task_marker[0],
                                proj_marker[0],
                                h[0],
                            ],
                            [
                                "base",
                                "task",
                                "proj",
                                r"policy ratio = $\alpha$",
                            ],
                        )

            for k, idx in enumerate(idxs):

                sigma_x_min = min(
                    self.std_base[idx][0], self.std_task[idx][0], self.std_proj[idx][0]
                )
                sigma_x_max = self.std_base[idx][0] - ETA

                sigma_y_min = min(
                    self.std_base[idx][1], self.std_task[idx][1], self.std_proj[idx][1]
                )
                sigma_y_max = self.std_base[idx][1] - ETA

                if plot_ref_task:
                    sigma_x_min = min(
                        sigma_x_min,
                        self.std_ref_task[idx][0],
                        self.std_ref_proj[idx][0],
                    )
                    sigma_y_min = min(
                        sigma_y_min,
                        self.std_ref_task[idx][1],
                        self.std_ref_proj[idx][1],
                    )

                sigma_x_min = max(sigma_x_min - border, ETA)
                sigma_y_min = max(sigma_y_min - border, ETA)

                sigma_x = np.linspace(sigma_x_min, sigma_x_max, Nx)
                sigma_y = np.linspace(sigma_y_min, sigma_y_max, Ny)

                policy_ratio = np.zeros((Ny, Nx))
                if plot_ref_task:
                    policy_ratio_ref = np.zeros((Ny, Nx))
                sigma = np.zeros(2)

                for i in range(0, Nx):
                    for j in range(0, Ny):
                        sigma[0] = sigma_x[i]
                        sigma[1] = sigma_y[j]
                        policy_ratio[j][i] = np.prod(
                            np.multiply(
                                self.std_base[k] / sigma,
                                np.exp(
                                    1
                                    / 2
                                    * np.square(self.mu_proj[k] - self.mu_base[k])
                                    / (np.square(self.std_base[k]) - np.square(sigma))
                                ),
                            )
                        )
                if plot_ref_task:
                    for i in range(0, Nx):
                        for j in range(0, Ny):
                            sigma[0] = sigma_x[i]
                            sigma[1] = sigma_y[j]
                            policy_ratio_ref[j][i] = np.prod(
                                np.multiply(
                                    self.std_base[k] / sigma,
                                    np.exp(
                                        1
                                        / 2
                                        * np.square(
                                            self.mu_ref_proj[k] - self.mu_base[k]
                                        )
                                        / (
                                            np.square(self.std_base[k])
                                            - np.square(sigma)
                                        )
                                    ),
                                )
                            )
                if plot_cmap:
                    im = axs[2 * k + 1].pcolormesh(sigma_x, sigma_y, policy_ratio)
                    # fig.colorbar(im, label="Policy Ratio")
                cs = axs[2 * k + 1].contour(
                    sigma_x,
                    sigma_y,
                    policy_ratio,
                    levels=extra_alphas + [self.alpha],
                    colors="k",
                )
                axs[2 * k + 1].clabel(cs, inline=True, fontsize=10)
                if plot_ref_task:
                    cs_ref = axs[2 * k + 1].contour(
                        sigma_x,
                        sigma_y,
                        policy_ratio_ref,
                        levels=extra_alphas + [self.alpha],
                        colors="k",
                        linestyles="--",
                    )
                    axs[2 * k + 1].clabel(cs_ref, inline=True, fontsize=10)
                axs[2 * k + 1].plot(
                    self.std_base[idx][0],
                    self.std_base[idx][1],
                    "rx",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="base",
                )
                axs[2 * k + 1].plot(
                    self.std_task[idx][0],
                    self.std_task[idx][1],
                    "bx",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="task",
                )
                axs[2 * k + 1].plot(
                    self.std_proj[idx][0],
                    self.std_proj[idx][1],
                    "m+",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="proj",
                )
                if plot_ref_task:
                    axs[2 * k + 1].plot(
                        self.std_ref_task[idx][0],
                        self.std_ref_task[idx][1],
                        "gx",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label="task (ref)",
                    )
                    axs[2 * k + 1].plot(
                        self.std_ref_proj[idx][0],
                        self.std_ref_proj[idx][1],
                        "y+",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label="proj (ref)",
                    )
                axs[2 * k + 1].set_xlabel(r"$\sigma_{0}$ (STD Forward Drive Force)")
                axs[2 * k + 1].set_ylabel(r"$\sigma_{1}$ (STD Turning Velocity)")

                if k == 0:
                    axs[2 * k + 1].set_title("Policy Standard Deviations")

        else:
            fig, axs = plt.subplots(
                1, N, figsize=(fig_size_scale_factor * N, fig_size_scale_factor)
            )

            for k, idx in enumerate(idxs):
                mu_x_min = min(
                    self.mu_base[idx][0], self.mu_task[idx][0], self.mu_proj[idx][0]
                )
                mu_x_max = max(
                    self.mu_base[idx][0], self.mu_task[idx][0], self.mu_proj[idx][0]
                )

                mu_y_min = min(
                    self.mu_base[idx][1], self.mu_task[idx][1], self.mu_proj[idx][1]
                )
                mu_y_max = max(
                    self.mu_base[idx][1], self.mu_task[idx][1], self.mu_proj[idx][1]
                )

                if plot_ref_task:
                    mu_x_min = min(
                        mu_x_min,
                        self.mu_ref_task[idx][0],
                        self.mu_ref_proj[idx][0],
                    )
                    mu_x_max = max(
                        mu_x_max,
                        self.mu_ref_task[idx][0],
                        self.mu_ref_proj[idx][0],
                    )
                    mu_y_min = min(
                        mu_y_min,
                        self.mu_ref_task[idx][1],
                        self.mu_ref_proj[idx][1],
                    )
                    mu_y_max = max(
                        mu_y_max,
                        self.mu_ref_task[idx][1],
                        self.mu_ref_proj[idx][1],
                    )

                mu_x_min -= border
                mu_x_max += border
                mu_y_min -= border
                mu_y_max += border

                mu_x = np.linspace(mu_x_min, mu_x_max, Nx)
                mu_y = np.linspace(mu_y_min, mu_y_max, Ny)

                policy_ratio = np.zeros((Ny, Nx))
                if plot_ref_task:
                    policy_ratio_ref = np.zeros((Ny, Nx))
                mu = np.zeros(2)

                for i in range(0, Nx):
                    for j in range(0, Ny):
                        mu[0] = mu_x[i]
                        mu[1] = mu_y[j]
                        policy_ratio[j][i] = np.prod(
                            np.multiply(
                                self.std_base[idx] / self.std_proj[idx],
                                np.exp(
                                    1
                                    / 2
                                    * np.square(mu - self.mu_base[idx])
                                    / (
                                        np.square(self.std_base[idx])
                                        - np.square(self.std_proj[idx])
                                    )
                                ),
                            )
                        )
                if plot_ref_task:
                    for i in range(0, Nx):
                        for j in range(0, Ny):
                            mu[0] = mu_x[i]
                            mu[1] = mu_y[j]
                            policy_ratio_ref[j][i] = np.prod(
                                np.multiply(
                                    self.std_base[idx] / self.std_ref_proj[idx],
                                    np.exp(
                                        1
                                        / 2
                                        * np.square(mu - self.mu_base[idx])
                                        / (
                                            np.square(self.std_base[idx])
                                            - np.square(self.std_ref_proj[idx])
                                        )
                                    ),
                                )
                            )
                if plot_cmap:
                    im = axs[k].pcolormesh(mu_x, mu_y, policy_ratio)
                    # fig.colorbar(im, label="Policy Ratio")
                cs = axs[k].contour(
                    mu_x,
                    mu_y,
                    policy_ratio,
                    levels=extra_alphas + [self.alpha],
                    colors="k",
                )
                h, l = cs.legend_elements()
                axs[k].clabel(cs, inline=True, fontsize=10)
                if plot_ref_task:
                    cs_ref = axs[k].contour(
                        mu_x,
                        mu_y,
                        policy_ratio_ref,
                        levels=extra_alphas + [self.alpha],
                        colors="k",
                        linestyles="--",
                    )
                    h_ref, l_ref = cs_ref.legend_elements()
                    axs[k].clabel(cs_ref, inline=True, fontsize=10)
                base_marker = axs[k].plot(
                    self.mu_base[idx][0],
                    self.mu_base[idx][1],
                    "rx",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="base",
                )
                task_marker = axs[k].plot(
                    self.mu_task[idx][0],
                    self.mu_task[idx][1],
                    "bx",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="task",
                )
                proj_marker = axs[k].plot(
                    self.mu_proj[idx][0],
                    self.mu_proj[idx][1],
                    "m+",
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label="proj",
                )
                if plot_ref_task:
                    task_ref_marker = axs[k].plot(
                        self.mu_ref_task[idx][0],
                        self.mu_ref_task[idx][1],
                        "gx",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label="task (ref)",
                    )
                    proj_ref_marker = axs[k].plot(
                        self.mu_ref_proj[idx][0],
                        self.mu_ref_proj[idx][1],
                        "y+",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label="proj (ref)",
                    )
                if k == 0:
                    axs[k].set_ylabel(r"$\mu_{1}$ (Mean Turning Velocity)")

                if k == 0:
                    if plot_ref_task:
                        fig.legend(
                            [
                                base_marker[0],
                                task_marker[0],
                                proj_marker[0],
                                task_ref_marker[0],
                                proj_ref_marker[0],
                                h[0],
                                h_ref[0],
                            ],
                            [
                                "base",
                                "task",
                                "proj",
                                "task (ref)",
                                "proj (ref)",
                                r"policy ratio = $\alpha$",
                                r"policy ratio = $\alpha$ (ref)",
                            ],
                            loc="outside lower right",
                            ncol=7,
                        )
                    else:
                        fig.legend(
                            [
                                base_marker[0],
                                task_marker[0],
                                proj_marker[0],
                                h[0],
                            ],
                            [
                                "base",
                                "task",
                                "proj",
                                r"policy ratio = $\alpha$",
                            ],
                            loc="outside lower right",
                            ncol=7,
                        )
            # fig.suptitle("Policy Means")
            fig.supxlabel(r"$\mu_{0}$ (Mean Forward Drive Force)")
            fig.tight_layout()
        return fig, axs

    def plot_policy_projection_with_frames(
        self,
        all_ep_idxs: Container[int],
        extracted_frames: list,
        timesteps: Container[int],
        plot_cmap: bool = False,
        Nx: int = 50,
        Ny: int = 50,
        border: float = 0.01,
        marker_size: int = 10,
        marker_edge_width: int = 2,
        extra_alphas: list[float] = [],
        fig_size_scale_factor: float = 7.0,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot policy projection data for the selected samples.

        Note that the contour and color map assumes the std is kept fixed.
        """
        assert self.num_collected_data > 0, "No data has been collected yet."
        assert not isinstance(self.mu_base, list), "Data has not been concatenated yet."
        assert (
            self.mu_base.shape[1] == 2
        ), "Only 2D policy projection plotting is supported."

        N = len(timesteps)

        idxs = all_ep_idxs[timesteps]

        fig, axs = plt.subplots(
            2, N, figsize=(fig_size_scale_factor * N, fig_size_scale_factor * 2)
        )
        axs = axs.ravel()
        for k, idx in enumerate(idxs):
            # Plot frame
            axs[k].imshow(extracted_frames[timesteps[k]])
            axs[k].set_title(f"t = {timesteps[k]}")
            axs[k].axis("off")

            # Plot policy projection
            mu_x_min = min(
                self.mu_base[idx][0], self.mu_task[idx][0], self.mu_proj[idx][0]
            )
            mu_x_max = max(
                self.mu_base[idx][0], self.mu_task[idx][0], self.mu_proj[idx][0]
            )

            mu_y_min = min(
                self.mu_base[idx][1], self.mu_task[idx][1], self.mu_proj[idx][1]
            )
            mu_y_max = max(
                self.mu_base[idx][1], self.mu_task[idx][1], self.mu_proj[idx][1]
            )

            mu_x_min -= border
            mu_x_max += border
            mu_y_min -= border
            mu_y_max += border

            mu_x = np.linspace(mu_x_min, mu_x_max, Nx)
            mu_y = np.linspace(mu_y_min, mu_y_max, Ny)

            policy_ratio = np.zeros((Ny, Nx))
            mu = np.zeros(2)

            for i in range(0, Nx):
                for j in range(0, Ny):
                    mu[0] = mu_x[i]
                    mu[1] = mu_y[j]
                    policy_ratio[j][i] = np.prod(
                        np.multiply(
                            self.std_base[idx] / self.std_proj[idx],
                            np.exp(
                                1
                                / 2
                                * np.square(mu - self.mu_base[idx])
                                / (
                                    np.square(self.std_base[idx])
                                    - np.square(self.std_proj[idx])
                                )
                            ),
                        )
                    )
            if plot_cmap:
                im = axs[k + N].pcolormesh(mu_x, mu_y, policy_ratio)
                # fig.colorbar(im, label="Policy Ratio")
            cs = axs[k + N].contour(
                mu_x,
                mu_y,
                policy_ratio,
                levels=extra_alphas + [self.alpha],
                colors="k",
            )
            h, l = cs.legend_elements()
            # axs[k + N].clabel(cs, inline=True, fontsize=12)
            base_marker = axs[k + N].plot(
                self.mu_base[idx][0],
                self.mu_base[idx][1],
                "rx",
                markersize=marker_size,
                markeredgewidth=marker_edge_width,
                label="base",
            )
            task_marker = axs[k + N].plot(
                self.mu_task[idx][0],
                self.mu_task[idx][1],
                "bx",
                markersize=marker_size,
                markeredgewidth=marker_edge_width,
                label="task",
            )
            proj_marker = axs[k + N].plot(
                self.mu_proj[idx][0],
                self.mu_proj[idx][1],
                marker="+",
                color="green",
                markersize=marker_size,
                markeredgewidth=marker_edge_width,
                label="proj",
                linestyle="None",
            )
            if k == 0:
                axs[k + N].set_ylabel(r"$\mu_{1}$ (Mean Turning Velocity)")
            axs[k + N].minorticks_on()
            if abs(self.mu_base[idx][1] - self.mu_task[idx][1]) < 0.4:
                axs[k + N].xaxis.set_major_locator(plt.MultipleLocator(0.1))
                axs[k + N].yaxis.set_major_locator(plt.MultipleLocator(0.1))
            axs[k + N].grid(which="major")
            axs[k + N].grid(which="minor", linestyle="--", alpha=0.5)
            if k == 0:
                fig.legend(
                    [
                        base_marker[0],
                        task_marker[0],
                        proj_marker[0],
                        h[0],
                    ],
                    [
                        r"$\pi_{base}$",
                        r"$\pi_{task}$",
                        r"$\pi_{proj}$",
                        r"policy ratio = $\alpha$",
                    ],
                    loc="outside lower right",
                    ncol=7,
                    columnspacing=0.8,  # Adjust this value as needed
                )
            # hack to get 'contour' to show up even if it's too thin
            ETA_CONTOUR = 1e-3
            if (
                abs(self.mu_base[idx][0] - self.mu_proj[idx][0]) < ETA_CONTOUR
                or abs(self.mu_base[idx][1] - self.mu_proj[idx][1]) < ETA_CONTOUR
            ) and (
                abs(self.mu_task[idx][0] - self.mu_proj[idx][0]) > ETA_CONTOUR
                and abs(self.mu_task[idx][1] - self.mu_proj[idx][1]) > ETA_CONTOUR
            ):
                axs[k + N].autoscale(False)
                diff_x = self.mu_base[idx][0] - self.mu_proj[idx][0]
                diff_y = self.mu_base[idx][1] - self.mu_proj[idx][1]
                axs[k + N].plot(
                    (self.mu_base[idx][0] + diff_x, self.mu_base[idx][0] - diff_x),
                    (self.mu_base[idx][1] + diff_y, self.mu_base[idx][1] - diff_y),
                    "k-",
                    zorder=1,
                )
        fig.supxlabel(r"$\mu_{0}$ (Mean Forward Drive Force)")
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axs


def save_policy_projection_database(
    db: PolicyProjectionDatabase, filename: str
) -> None:
    """Save a dictionary to a file."""
    assert ".pkl" in filename, "filename must have .pkl extension."
    with open(filename, "wb") as file:
        pickle.dump(db, file, pickle.HIGHEST_PROTOCOL)


def load_policy_projection_database(filename: str) -> PolicyProjectionDatabase:
    """Load a dictionary from a file."""
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
    posterior_bound_failure_rate[alpha] = scenario_db.get_selected_epsilons([T], conf)


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
    task_mean_successful_time_taken: dict[float, float] | None = None,
    task_std_successful_time_taken: dict[float, float] | None = None,
    bounds_with_alphas: dict[float, float] | None = None,
    conf: float | None = None,
    bound_label_x_offset: float = 0.01,
    bound_label_y_offset: float = 0.01,
    fig_size_scale_factor: float = 1.0,
) -> tuple[Figure, Axes]:
    """
    Plot the mean and standard deviation of episode length for successful scenarios across different alphas.

    Optionally, plot the same for task policies trained under the alpha constraint, and vertical lines representing prior bound levels.
    """
    # Extract alpha values and corresponding means and standard deviations for successful time taken
    alphas = np.array(list(mean_successful_time_taken.keys()))
    means = np.array(list(mean_successful_time_taken.values()))
    stds = np.array(list(std_successful_time_taken.values()))

    # Plot the mean with standard deviations
    fig, ax = plt.subplots(
        figsize=(10 * fig_size_scale_factor, 5 * fig_size_scale_factor)
    )
    ax.plot(alphas, means, color="blue", marker="x", label=r"Pre-trained $\pi_{task}$")
    ax.fill_between(alphas, means - stds, means + stds, alpha=0.2, color="blue")
    ax.fill_between(
        alphas,
        means - stds,
        means + stds,
        alpha=1.0,
        facecolor="none",
        edgecolor="blue",
    )
    if (
        task_mean_successful_time_taken is not None
        and task_std_successful_time_taken is not None
    ):
        task_means = np.array(list(task_mean_successful_time_taken.values()))
        task_stds = np.array(list(task_std_successful_time_taken.values()))
        ax.plot(
            alphas,
            task_means,
            color="green",
            marker="x",
            label=r"Projected PPO",
        )
        ax.fill_between(
            alphas,
            task_means - task_stds,
            task_means + task_stds,
            alpha=0.2,
            color="green",
        )
        ax.fill_between(
            alphas,
            task_means - task_stds,
            task_means + task_stds,
            alpha=1.0,
            facecolor="none",
            edgecolor="green",
        )
        ax.legend()
    if bounds_with_alphas is not None and conf is not None:
        flag = True
        for bound, alpha in bounds_with_alphas.items():
            if flag:
                lbl = r"Prior bound $\epsilon_{task} = \epsilon_{base} \alpha^{{T}}$"
                flag = False
            else:
                lbl = "_nolegend_"
            ax.axvline(x=alpha, color="red", linestyle="--", label=lbl)
            ax.text(
                alpha + bound_label_x_offset,
                min(np.min(means - stds), np.min(task_means - task_stds))
                + bound_label_y_offset,
                str(bound),
                rotation=90,
                color="red",
            )
        ax.legend(loc="best")

    ax.set_xlim([alphas.min(), alphas.max()])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Mean \xb1 STD Episode Length on Success       ")
    ax.set_xscale("log")
    ax.minorticks_on()
    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    # hack to remove scientific notation when plotting close-up
    if alphas.max() < 10.0:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
    fig.tight_layout()
    return fig, ax


def plot_failure_probs(
    epsilon: float,
    T: int,
    empirical_failure_rate: dict[float, float],
    posterior_bound_failure_rate: dict[float, float],
    task_empirical_failure_rate: dict[float, float] | None = None,
    task_posterior_bound_failure_rate: dict[float, float] | None = None,
    conf: float = 0.9999999,
    fig_size_scale_factor: float = 1.0,
) -> tuple[Figure, Axes]:
    """
    Plot the empirical failure rate and prior/posterior bound on failure probability across different alphas.

    Optionally, plot the same for task policies trained under the alpha constraint.
    """
    # Extract alpha values and corresponding means and standard deviations for successful time taken
    alphas = np.array(list(empirical_failure_rate.keys()))
    failure_rates = np.array(list(empirical_failure_rate.values()))
    posterior_bounds = np.array(list(posterior_bound_failure_rate.values()))

    fig, ax = plt.subplots(
        figsize=(10 * fig_size_scale_factor, 5 * fig_size_scale_factor)
    )

    scen_str = r"Prior bound $\epsilon_{task} = \epsilon_{base} \alpha^{T}$"
    # alpha_range = np.linspace(alphas.min(), alphas.max(), 1000)
    alpha_range = np.logspace(np.log10(alphas.min()), np.log10(alphas.max()), 1000)
    prior_bounds = np.minimum(epsilon * alpha_range**T, 1.0)
    ax.plot(alpha_range, prior_bounds, color="red", label=scen_str, linestyle="--")

    emp_str = r"Violation rate ($\frac{{k}}{{N}}$) (pre-trained $\pi_{task}$)"
    ax.plot(alphas, failure_rates, color="blue", label=emp_str, marker="x")

    scen_str = r"Posterior bound (pre-trained $\pi_{task}$)"
    ax.plot(
        alphas,
        posterior_bounds,
        color="blue",
        label=scen_str,
        marker="x",
        linestyle="--",
    )

    if (
        task_empirical_failure_rate is not None
        and task_posterior_bound_failure_rate is not None
    ):
        task_failure_rates = np.array(list(task_empirical_failure_rate.values()))
        task_posterior_bounds = np.array(
            list(task_posterior_bound_failure_rate.values())
        )

        task_emp_str = r"Violation rate ($\frac{{k}}{{N}}$) (Projected PPO)"
        ax.plot(
            alphas,
            task_failure_rates,
            color="green",
            label=task_emp_str,
            marker="x",
        )

        task_scen_str = r"Posterior bound (Projected PPO)"
        ax.plot(
            alphas,
            task_posterior_bounds,
            color="green",
            label=task_scen_str,
            marker="x",
            linestyle="--",
        )

    ax.set_xlim([alphas.min(), alphas.max()])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Violation Probability")
    ax.set_xscale("log")
    ax.minorticks_on()
    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    if alphas.max() > 10:
        ax.legend(loc="upper center")
    else:
        ax.legend(loc="best")
    # hack to remove scientific notation when plotting close-up
    if alphas.max() < 10.0:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
    return fig, ax


def plot_max_log_alpha(
    scenario_db: ScenarioDatabase,
    bounds: Container[float],
    conf: float = 0.9999999,
    cutoff: float = 1.0,
    fig_size_scale_factor: float = 1.0,
) -> tuple[Figure, Axes, dict[float, tuple[int, float]]]:
    """
    LEGACY FUNCTION FOR OLD SCENARIO DATABASES -- now a method of ScenarioDatabase

    Plot the maximum permitted log-alpha over all possible maximum episode lengths T for each specified failure bound.

    Also returns the locations of the peaks for each bound (i.e. the T/log-alpha pair that permits the largest alpha for the bound).
    """
    fig, ax = plt.subplots(
        figsize=(10 * fig_size_scale_factor, 5 * fig_size_scale_factor)
    )
    peaks = {}
    for bound in reversed(bounds):
        log_alpha_array = np.maximum(
            (
                np.log(bound)
                - np.log(scenario_db.get_all_epsilons(conf=conf, cutoff=cutoff)[1:])
            )
            / np.arange(1, scenario_db.max_episode_length + 1),
            0.0,
        )  # alpha >= 1.0 required
        label = r"$\epsilon_{{base}}(T)\alpha^{{T}}={bound}$".format(bound=bound)
        ax.plot(
            np.arange(1, scenario_db.max_episode_length + 1),
            log_alpha_array,
            label=label,
        )

        peaks[bound] = (np.argmax(log_alpha_array) + 1, np.max(log_alpha_array))

    ax.set_xlim([1, scenario_db.max_episode_length + 1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Maximum Episode Length, T (time steps)")
    ax.set_ylabel(r"Maximum Permitted log($\alpha$)")
    ax.minorticks_on()
    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_title(r"$\beta={b:.1E}$".format(b=1 - conf))

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
