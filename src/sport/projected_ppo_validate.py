import os
import random
from logging import warning

import gymnasium as gym  # noqa: F401
import hydra
import numpy as np
import safety_gymnasium
import torch
import torch.nn as nn  # noqa: F401
import wandb
from gymnasium.spaces import Box
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sport.rl.algos.projected_ppo.core import MLPProjectedActorCritic
from sport.rl.algos.projected_ppo.dataclasses import ProjectedPPOValidateConfig
from sport.rl.utils import get_actor_structure, make_env_safety, process_info
from sport.validate.utils import (
    PolicyProjectionDatabase,
    ScenarioDatabase,
    load_policy_projection_database,
    load_scenario_database,
    save_policy_projection_database,
    save_scenario_database,
)


# Need to run everything inside hydra main function
@hydra.main(
    config_path="config", config_name="projected_ppo_validate", version_base="1.3"
)
def main(cfg: ProjectedPPOValidateConfig) -> None:
    # Print the config
    print(OmegaConf.to_yaml(cfg))

    # Note: hydra sweeping doesn't work very well with wandb logging
    # It is instead best to use wandb sweeping
    # I've included the functionality for hydra sweeping anyway

    # Get name for each job
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        run_name = HydraConfig.get().run.dir + "/" + str(HydraConfig.get().job.num)
    else:
        run_name = HydraConfig.get().run.dir

    working_dir = os.getcwd()
    original_dir = hydra.utils.get_original_cwd()
    print(f"Working directory : {working_dir}")

    alpha_str = str(float(cfg.alpha)).replace(".", "-")

    # print(f"Test : {run_name}")

    # Convert config into format suitable for wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Set up wandb logger
    if cfg.wandb.track:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            sync_tensorboard=True,
            config=wandb.config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            # settings=wandb.Settings(start_method="thread"),
        )

    # Set up tensorboard summary writer
    writer = SummaryWriter("log")

    if cfg.validate_common.control_rng:
        assert cfg.validate_common.num_envs == 1, "Controlled RNG requires num_envs=1"
    seed = cfg.validate_common.seed

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = cfg.validate_common.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.validate_common.cuda else "cpu"
    )

    # Environment setup
    # Note: vectorized envs
    if cfg.validate_common.env_seed == "None":  # null becomes "None" str in wandb sweep
        env_seed = None
    elif cfg.validate_common.env_seed is not None:
        assert isinstance(
            cfg.validate_common.env_seed, int
        ), "env_seed must be type int"
        env_seed = cfg.validate_common.env_seed
    prefix = "_alpha_task_" if cfg.use_alpha_task else "_alpha_"
    envs = safety_gymnasium.vector.SafetySyncVectorEnv(
        [
            make_env_safety(
                cfg.validate_common.gym_id,
                i,
                seed + i,
                cfg.validate_common.capture_video,
                cfg.validate_common.capture_video_ep_interval,
                cfg.validate_common.clip_action,
                cfg.validate_common.normalize_observation,
                cfg.validate_common.normalize_reward,
                video_dir=prefix + alpha_str,
                env_seed=env_seed,
                camera_name=cfg.validate_common.camera_name,
            )
            for i in range(cfg.validate_common.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, Box
    ), "only continuous action space is supported"

    # SafetySyncVectorEnv doesn't properly save the max_episode_steps in its spec
    # So we instead create a throwaway env object to get the max_episode_steps
    _ = safety_gymnasium.make(cfg.validate_common.gym_id)
    max_episode_steps = _.spec.max_episode_steps

    # Instantiate scenario database
    scenario_db = ScenarioDatabase(
        cfg.validate_common.num_envs,
        max_episode_steps,
        cfg.validate_common.total_eps,
    )
    if cfg.validate_common.load_db:
        # Load existing scenario database
        abs_db_path = os.path.join(original_dir, cfg.validate_common.load_db_path)
        if os.path.exists(abs_db_path):
            scenario_db = load_scenario_database(abs_db_path)
            assert (
                scenario_db.num_envs == cfg.validate_common.num_envs
            ), "Number of environments in scenario database ({a}) does not match number of environments in config ({b})".format(
                a=scenario_db.num_envs, b=cfg.validate_common.num_envs
            )
            assert (
                scenario_db.max_episode_length == max_episode_steps
            ), "Max episode length in scenario database ({a}) does not match max env episode length ({b})".format(
                a=scenario_db.max_episode_length, b=max_episode_steps
            )
            assert (
                scenario_db.max_num_scenarios == cfg.validate_common.total_eps
            ), "Total number of episodes in scenario database ({a}) does not match total number of episodes in config ({b})".format(
                a=scenario_db.max_num_scenarios, b=cfg.validate_common.total_eps
            )
            scenario_db.reset_active_scenarios()
        else:
            warning(
                "Scenario database path {f} does not exist. Creating new scenario database.".format(
                    f=abs_db_path
                )
            )

    # Instantiate policy projection database
    policy_projection_db = PolicyProjectionDatabase(
        alpha=cfg.alpha,
        check_ref_task_policy=cfg.check_ref_task_policy,
    )
    if cfg.load_policy_projection_db:
        # Load existing policy projection database
        abs_policy_projection_db_path = os.path.join(
            original_dir, cfg.load_policy_projection_db_path
        )
        if os.path.exists(abs_policy_projection_db_path):
            policy_projection_db = load_policy_projection_database(
                abs_policy_projection_db_path
            )
            assert (
                policy_projection_db.check_ref_task_policy == cfg.check_ref_task_policy
            ), "check_ref_task_policy flag in policy projection database does not match config"
        else:
            warning(
                "Policy projection database path {f} does not exist. Creating new policy projection database.".format(
                    f=abs_policy_projection_db_path
                )
            )

    # Load policy state dicts
    abs_base_policy_path = os.path.join(original_dir, cfg.validate_common.policy_path)
    assert os.path.exists(
        abs_base_policy_path
    ), "Base policy path {path} does not exist".format(path=abs_base_policy_path)
    base_loaded_state_dict = torch.load(
        abs_base_policy_path, weights_only=True, map_location=device
    )

    if cfg.use_alpha_task:
        policy_dir, policy_file = os.path.split(cfg.task_policy_path)
        alpha_task_policy_path = policy_dir + "/alpha_" + alpha_str + "_" + policy_file
        abs_task_policy_path = os.path.join(original_dir, alpha_task_policy_path)
    else:
        abs_task_policy_path = os.path.join(original_dir, cfg.task_policy_path)
    assert os.path.exists(
        abs_task_policy_path
    ), "Task policy path {path} does not exist".format(path=abs_task_policy_path)
    task_loaded_state_dict = torch.load(
        abs_task_policy_path, weights_only=True, map_location=device
    )

    if cfg.check_ref_task_policy:
        abs_ref_task_policy_path = os.path.join(original_dir, cfg.ref_task_policy_path)
        assert os.path.exists(
            abs_ref_task_policy_path
        ), "Task policy path {path} does not exist".format(
            path=abs_ref_task_policy_path
        )
        ref_task_loaded_state_dict = torch.load(
            abs_ref_task_policy_path, weights_only=True, map_location=device
        )

    # Construct agent from state dicts
    base_loaded_hidden_sizes, base_loaded_activation = get_actor_structure(
        base_loaded_state_dict, envs.single_observation_space, envs.single_action_space
    )
    task_loaded_hidden_sizes, task_loaded_activation = get_actor_structure(
        task_loaded_state_dict, envs.single_observation_space, envs.single_action_space
    )
    if cfg.check_ref_task_policy:
        ref_task_loaded_hidden_sizes, ref_task_loaded_activation = get_actor_structure(
            ref_task_loaded_state_dict,
            envs.single_observation_space,
            envs.single_action_space,
        )
    else:
        ref_task_loaded_hidden_sizes, ref_task_loaded_activation = get_actor_structure(
            task_loaded_state_dict,
            envs.single_observation_space,
            envs.single_action_space,
        )

    agent = MLPProjectedActorCritic(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        hidden_sizes_base=base_loaded_hidden_sizes,
        activation_base=eval("nn." + base_loaded_activation + "()"),
        hidden_sizes_task=task_loaded_hidden_sizes,
        activation_task=eval("nn." + task_loaded_activation + "()"),
        alpha=cfg.alpha,
        check_max_policy_ratios=cfg.check_max_policy_ratios,
        check_ref_task_policy=cfg.check_ref_task_policy,
        hidden_sizes_ref_task=ref_task_loaded_hidden_sizes,
        activation_ref_task=eval("nn." + ref_task_loaded_activation + "()"),
    )
    agent.pi_base.load_state_dict(base_loaded_state_dict, strict=True)
    agent.pi_task.load_state_dict(task_loaded_state_dict, strict=True)
    if cfg.check_ref_task_policy:
        agent.pi_ref_task.load_state_dict(ref_task_loaded_state_dict, strict=True)
    agent.to(device)

    goal_achieved_count = 0
    constraint_violated_count = 0

    # Prevent storing gradients
    for p in agent.parameters():
        p.requires_grad = False

    # Initialize validation loop
    global_step = 0
    if env_seed is not None:
        obs, info = envs.reset()  # env seed is already set using the SeedWrapper
    else:
        obs, info = envs.reset(seed=[seed + i for i in range(scenario_db.num_envs)])
    done = np.full(scenario_db.num_envs, False, dtype=bool)
    goal_achieved = np.full(scenario_db.num_envs, False, dtype=bool)
    constraint_violated = np.full(scenario_db.num_envs, False, dtype=bool)
    scenario_db.update(done, goal_achieved, constraint_violated)
    writer.add_scalar(
        "charts/num_collected_scenarios",
        scenario_db.num_collected_scenarios,
        global_step=global_step,
    )
    writer.add_scalar(
        "charts/goal_achieved_count",
        goal_achieved_count,
        global_step=global_step,
    )
    writer.add_scalar(
        "charts/constraint_violated_count",
        constraint_violated_count,
        global_step=global_step,
    )

    episodic_length = np.zeros(scenario_db.num_envs, dtype=int)

    # Episode counter
    # To be consistent with episode tracking for videos, track episode of each env separately
    episode_count = np.array([0] * cfg.validate_common.num_envs, dtype=int)

    # Instantiate progress bar
    pbar = tqdm(
        total=(scenario_db.max_num_scenarios - scenario_db.num_collected_scenarios)
    )
    # VALIDATION LOOP:
    with torch.no_grad():  # no gradient needed for testing
        while scenario_db.num_collected_scenarios < scenario_db.max_num_scenarios:

            act = agent.act(torch.Tensor(obs).to(device))

            policy_projection_db.update(agent, episode_count)

            if cfg.check_max_policy_ratios:
                for i in range(cfg.validate_common.num_envs):
                    # note: log_task_base may contain infinities!
                    writer.add_scalar(
                        "charts/task_max_log_policy_ratio",
                        agent.latest_log_task_base[i],
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "charts/proj_max_log_policy_ratio",
                        agent.latest_log_proj_base[i],
                        global_step=global_step,
                    )
                    if cfg.check_ref_task_policy:
                        writer.add_scalar(
                            "charts/ref_task_max_log_policy_ratio",
                            agent.latest_log_ref_task_base[i],
                            global_step=global_step,
                        )
                        writer.add_scalar(
                            "charts/ref_proj_max_log_policy_ratio",
                            agent.latest_log_ref_proj_base[i],
                            global_step=global_step,
                        )

            if cfg.check_ref_task_policy:
                for i in range(cfg.validate_common.num_envs):
                    writer.add_scalar(
                        "charts/task_to_ref_kl_divergence",
                        policy_projection_db.kl_div_ref_task[-1][i],
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "charts/proj_to_ref_kl_divergence",
                        policy_projection_db.kl_div_ref_proj[-1][i],
                        global_step=global_step,
                    )

            next_obs, _, _, term, trunc, info = envs.step(act.detach().cpu().numpy())

            episodic_length += 1

            done = np.logical_or(term, trunc)
            goal_achieved, constraint_violated = process_info(info)

            obs = next_obs
            previous_active_scenarios = scenario_db.active_scenarios.copy()
            previous_num_collected_scenarios = scenario_db.num_collected_scenarios

            scenario_db.update(done, goal_achieved, constraint_violated)

            global_step += 1 * scenario_db.num_envs

            if scenario_db.num_collected_scenarios > previous_num_collected_scenarios:
                writer.add_scalar(
                    "charts/num_collected_scenarios",
                    scenario_db.num_collected_scenarios,
                    global_step=global_step,
                )
                goal_achieved_count += np.sum(goal_achieved)
                writer.add_scalar(
                    "charts/goal_achieved_count",
                    goal_achieved_count,
                    global_step=global_step,
                )
                constraint_violated_count += np.sum(constraint_violated)
                writer.add_scalar(
                    "charts/constraint_violated_count",
                    constraint_violated_count,
                    global_step=global_step,
                )

            for i in range(scenario_db.num_envs):
                if done[i]:
                    writer.add_scalar(
                        "charts/episodic_length",
                        episodic_length[i],
                        global_step=global_step,
                    )
                    episodic_length[i] = 0

            max_num_scenarios_complete = sum(
                previous_active_scenarios != scenario_db.active_scenarios
            )

            # Increment episode counter
            episode_count += done

            # Save scenario database and policy projection database
            if cfg.validate_common.save_db:
                if (
                    scenario_db.num_collected_scenarios
                    % cfg.validate_common.save_db_ep_interval
                    == 0
                ):
                    prefix = "alpha_task_" if cfg.use_alpha_task else "alpha_"
                    save_scenario_database(
                        scenario_db, prefix + alpha_str + "_scenario_db.pkl"
                    )
                    save_policy_projection_database(
                        policy_projection_db,
                        prefix + alpha_str + "_projection_db.pkl",
                    )

            pbar.update(max_num_scenarios_complete)

            if cfg.validate_common.control_rng:
                # Reset RNG
                if np.any(done):
                    # Increment seed
                    seed += 1
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    envs.action_space.seed(seed)
                    envs.observation_space.seed(seed)
                    episodic_length = np.zeros(scenario_db.num_envs, dtype=int)

    # Close progress bar
    pbar.close()

    # Save final scenario database and policy projection database
    if cfg.validate_common.save_db:
        prefix = "alpha_task_" if cfg.use_alpha_task else "alpha_"
        save_scenario_database(scenario_db, prefix + alpha_str + "_scenario_db.pkl")
        save_policy_projection_database(
            policy_projection_db, prefix + alpha_str + "_projection_db.pkl"
        )

    # Close envs
    envs.close()

    # Close summary writer
    writer.close()

    # TODO: perhaps add option to plot the epsilons at the end?
    # But the user would be able to just load the scenario database and plot it themselves


if __name__ == "__main__":
    main()
