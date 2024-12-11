import os
import random

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

from rl_vcf.rl.algos.sac.core import MLPActorCritic
from rl_vcf.rl.utils import get_actor_structure, make_env_safety, process_info
from rl_vcf.validate.dataclasses import ValidateConfig
from rl_vcf.validate.utils import (
    ScenarioDatabase,
    load_scenario_database,
    save_scenario_database,
)


# Need to run everything inside hydra main function
@hydra.main(config_path="config", config_name="sac_safety_validate", version_base="1.3")
def main(cfg: ValidateConfig) -> None:
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
    # writer.add_text(
    #     "config",
    #     "|param|value|\n|-|-|\n%s"
    #     % (
    #         "\n".join(
    #             [f"|{key}|{value}|" for key, value in cfg.train_common.items()]
    #             + [f"|{key}|{value}|" for key, value in cfg.train.items()]
    #             + [f"|{key}|{value}|" for key, value in cfg.network.items()]
    #         )
    #     ),
    # )

    # Seeding
    random.seed(cfg.validate_common.seed)
    np.random.seed(cfg.validate_common.seed)
    torch.manual_seed(cfg.validate_common.seed)
    torch.backends.cudnn.deterministic = cfg.validate_common.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.validate_common.cuda else "cpu"
    )

    # Environment setup
    # Note: vectorized envs
    envs = safety_gymnasium.vector.SafetySyncVectorEnv(
        [
            make_env_safety(
                cfg.validate_common.gym_id,
                i,
                cfg.validate_common.seed + i,
                cfg.validate_common.capture_video,
                cfg.validate_common.capture_video_ep_interval,
                cfg.validate_common.preprocess_envs,
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

    if cfg.validate_common.load_db:
        # Load existing scenario database
        abs_db_path = os.path.join(original_dir, cfg.validate_common.load_db_path)
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
        scenario_db = ScenarioDatabase(
            cfg.validate_common.num_envs,
            max_episode_steps,
            cfg.validate_common.total_eps,
        )

    # Load policy state dict
    abs_policy_path = os.path.join(original_dir, cfg.validate_common.policy_path)
    assert os.path.exists(abs_policy_path), "Policy path {path} does not exist".format(
        path=abs_policy_path
    )
    loaded_state_dict = torch.load(
        abs_policy_path, weights_only=True, map_location=device
    )

    # Construct agent from state dict
    loaded_hidden_sizes, loaded_activation = get_actor_structure(
        loaded_state_dict, envs.single_observation_space, envs.single_action_space
    )

    agent = MLPActorCritic(
        envs.single_observation_space,
        envs.single_action_space,
        loaded_hidden_sizes,
        eval("nn." + loaded_activation + "()"),
    )
    agent.pi.load_state_dict(loaded_state_dict, strict=True)
    agent.to(device)

    # Prevent storing gradients
    for p in agent.parameters():
        p.requires_grad = False

    # Initialize validation loop
    global_step = 0
    obs, info = envs.reset(
        seed=[cfg.validate_common.seed + i for i in range(scenario_db.num_envs)]
    )
    done = np.full(scenario_db.num_envs, False, dtype=bool)
    goal_achieved = np.full(scenario_db.num_envs, False, dtype=bool)
    constraint_violated = np.full(scenario_db.num_envs, False, dtype=bool)
    scenario_db.update(done, goal_achieved, constraint_violated)
    writer.add_scalar(
        "charts/num_collected_scenarios",
        scenario_db.num_collected_scenarios,
        global_step=global_step,
    )

    # Instantiate progress bar
    pbar = tqdm(
        total=(scenario_db.max_num_scenarios - scenario_db.num_collected_scenarios)
    )
    # VALIDATION LOOP:
    with torch.no_grad():  # no gradient needed for testing
        while scenario_db.num_collected_scenarios < scenario_db.max_num_scenarios:

            act = agent.act(torch.Tensor(obs).to(device))

            next_obs, _, _, term, trunc, info = envs.step(act.detach().cpu().numpy())

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

            max_num_scenarios_complete = sum(
                previous_active_scenarios != scenario_db.active_scenarios
            )

            # Save scenario database
            if cfg.validate_common.save_db:
                if (
                    scenario_db.num_collected_scenarios
                    % cfg.validate_common.save_db_ep_interval
                    == 0
                ):
                    save_scenario_database(scenario_db, "base_policy_db.pkl")

            pbar.update(max_num_scenarios_complete)

    # Close progress bar
    pbar.close()

    # Save final scenario database
    if cfg.validate_common.save_db:
        save_scenario_database(scenario_db, "base_policy_db.pkl")

    # Close envs
    envs.close()

    # Close summary writer
    writer.close()

    # TODO: perhaps add option to plot the epsilons at the end?
    # But the user would be able to just load the scenario database and plot it themselves


if __name__ == "__main__":
    main()
