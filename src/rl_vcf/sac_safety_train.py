import itertools
import logging
import os
import random
import time
from copy import deepcopy

import gymnasium as gym  # noqa: F401
import hydra
import numpy as np
import safety_gymnasium
import torch
import torch.nn as nn  # noqa: F401
import torch.optim as optim
import wandb
from gymnasium.spaces import Box
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_vcf.rl.algos.sac.core import MLPActorCritic, ReplayBuffer
from rl_vcf.rl.algos.sac.dataclasses import SACSafetyConfig
from rl_vcf.rl.utils import make_env_safety


# Need to run everything inside hydra main function
@hydra.main(config_path="config", config_name="sac_safety", version_base="1.3")
def main(cfg: SACSafetyConfig) -> None:
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
    writer.add_text(
        "config",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [f"|{key}|{value}|" for key, value in cfg.train_common.items()]
                + [f"|{key}|{value}|" for key, value in cfg.train.items()]
                + [f"|{key}|{value}|" for key, value in cfg.network.items()]
            )
        ),
    )

    # Seeding
    random.seed(cfg.train_common.seed)
    np.random.seed(cfg.train_common.seed)
    torch.manual_seed(cfg.train_common.seed)
    torch.backends.cudnn.deterministic = cfg.train_common.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.train_common.cuda else "cpu"
    )

    # Environment setup
    # Note: vectorized envs
    envs = safety_gymnasium.vector.SafetySyncVectorEnv(
        [
            make_env_safety(
                cfg.train_common.gym_id,
                i,
                cfg.train_common.seed + i,
                cfg.train_common.capture_video,
                cfg.train_common.capture_video_ep_interval,
                cfg.train_common.preprocess_envs,
            )
            for i in range(cfg.train_common.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, Box
    ), "only continuous action space is supported"

    # Episode counter
    # To be consistent with episode tracking for videos, track episode of each env separately
    episode_count = np.array([0] * cfg.train_common.num_envs, dtype=int)

    if cfg.train_common.save_model:
        models_folder = os.path.abspath("models")
        # Create models output folder if needed
        if os.path.isdir(models_folder):
            logging.warning(f"Overwriting existing models at {models_folder} folder")
        os.makedirs(models_folder, exist_ok=True)
        policies_folder = os.path.abspath("policies")
        # Create policies output folder if needed
        if os.path.isdir(policies_folder):
            logging.warning(
                f"Overwriting existing policies at {policies_folder} folder"
            )
        os.makedirs(policies_folder, exist_ok=True)

    # Create agent
    agent = MLPActorCritic(
        envs.single_observation_space,
        envs.single_action_space,
        cfg.network.hidden_sizes,
        eval("nn." + cfg.network.activation + "()"),
    ).to(device)
    print(agent)

    # Create target networks
    agent_targ = deepcopy(agent)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in agent_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q networks (for convenience)
    q_params = itertools.chain(agent.q1.parameters(), agent.q2.parameters())

    # Create optimizers
    policy_optimizer = optim.Adam(agent.pi.parameters(), lr=cfg.train.policy_lr)
    q_optimizer = optim.Adam(q_params, lr=cfg.train.q_lr)

    # Automatic entropy tuning
    if cfg.train.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_ent_coeff = torch.zeros(1, requires_grad=True, device=device)
        ent_coeff = log_ent_coeff.exp().item()
        ent_coeff_optimizer = optim.Adam([log_ent_coeff], lr=cfg.train.q_lr)
    else:
        ent_coeff = cfg.train.ent_coef

    # Not sure why this is in the cleanrl code
    # including it causes warnings (gym output is np.float64)
    # envs.single_observation_space.dtype = np.float32

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        cfg.train.buffer_size,
        device,
    )

    # Track whether the goal was achieved and whether a constraint was violated each episode
    goal_achieved = False
    goal_achieved_latch = False
    constraint_violated = False
    constraint_violated_latch = False

    # Track how many times the goal was achieved each episode
    goal_achieved_count = 0

    # Start training
    start_time = time.time()
    obs = torch.Tensor(
        envs.reset(
            seed=[cfg.train_common.seed + i for i in range(cfg.train_common.num_envs)]
        )[
            0  # observations are first element of env reset output
        ]
    ).to(device)
    for global_step in tqdm(range(cfg.train_common.total_timesteps)):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            if global_step < cfg.train.burn_in:
                act = torch.Tensor(
                    np.array(
                        [
                            envs.single_action_space.sample()
                            for _ in range(cfg.train_common.num_envs)
                        ]
                    )
                )
            else:
                act, _ = agent.pi.forward(obs, with_log_prob=False)

        # Environment step
        next_obs, rew, cost, term, trunc, info = envs.step(act.detach().cpu().numpy())

        # Update latches
        if term | trunc:
            # note that normal info goes to "final_info" at the end of the episode
            for item in info["final_info"]:
                if item and "goal_met" in item:
                    goal_achieved = item["goal_met"]
                if item and "constraint_violated" in item:
                    constraint_violated = item["constraint_violated"]
        else:
            goal_achieved = info["goal_met"]
            constraint_violated = info["constraint_violated"]

        if goal_achieved:
            goal_achieved_count += 1
        goal_achieved_latch = goal_achieved_latch | goal_achieved
        constraint_violated_latch = constraint_violated_latch | constraint_violated

        # Learn safety
        # TODO: improve safety learning
        # TODO: support parallelization
        if cfg.learn_safety:
            rew = rew * 0.1  # dense reward (Euclidean distance)

            if goal_achieved:
                rew -= rew
                rew += 1.0  # sparse reward
            if term | trunc:
                if (not goal_achieved_latch) | constraint_violated_latch:
                    rew -= rew
                    # rew = -cost
                    rew += -1.0  # sparse penalty

        # Record episodic returns for plotting
        if "final_info" in info:
            # note info is usually empty but gets populated at end of episode
            # also, the current gymnasium implementation of info seems absolutely terrible?
            # a dict of lists, each list containing a mix of different types, including dicts??
            for item in info["final_info"]:
                if item and "episode" in item:
                    # print(
                    #     f"global_step={global_step}, episodic_return={item['episode']['r']}"
                    # )
                    writer.add_scalar(
                        "charts/episodic_return",
                        item["episode"]["r"],
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        item["episode"]["l"],
                        global_step=global_step,
                    )
            writer.add_scalar(
                "charts/goal_achieved", goal_achieved_latch, global_step=global_step
            )
            writer.add_scalar(
                "charts/constraint_violated",
                constraint_violated_latch,
                global_step=global_step,
            )
            writer.add_scalar(
                "charts/task_success",
                goal_achieved_latch & (not constraint_violated_latch),
                global_step=global_step,
            )
            writer.add_scalar(
                "charts/goal_achieved_count",
                goal_achieved_count,
                global_step=global_step,
            )

        # Reset latches if at end of episode
        if term | trunc:
            goal_achieved_latch = False
            constraint_violated_latch = False
            goal_achieved_count = 0

        done = np.logical_or(term, trunc)
        if done[0]:
            if cfg.train_common.save_model:
                # To be consistent with episode tracking for videos,
                # use episode number of env 0 (not total episode count!)
                if episode_count[0] % cfg.train_common.save_model_ep_interval == 0:
                    # Save model
                    model_name = f"models/rl-model-episode-{episode_count[0]}.pt"
                    torch.save(agent.state_dict(), model_name)
                    # Save policy separately as well
                    policy_name = f"policies/rl-policy-episode-{episode_count[0]}.pt"
                    torch.save(agent.pi.state_dict(), policy_name)

            writer.add_scalar(
                "charts/episode_count_per_env",
                episode_count[0],
                global_step=global_step,
            )
        # Increment episode counter
        episode_count += done

        # Handle "final_observation"
        # Note that for vectorized environments, each environment auto-resets when done
        # So when we step, we actually see the initial observation of the next episode
        # So if we want the final observation of the previous episode, we look in info
        real_next_obs = next_obs.copy()
        for idx, truncated in enumerate(trunc):
            if truncated:
                real_next_obs[idx] = info["final_observation"][idx]

        # Save data to replay buffer
        replay_buffer.store(
            obs,
            act,
            torch.Tensor(rew).to(device),
            torch.Tensor(real_next_obs).to(device),
            torch.Tensor(term).to(device),
        )

        # Update obs (critical!)
        obs = torch.Tensor(next_obs).to(device)

        # ALGO LOGIC: training
        if global_step > cfg.train.burn_in:
            # Q networks update
            data = replay_buffer.sample_batch(cfg.train.batch_size)
            with torch.no_grad():
                next_pi, next_log_pi = agent.pi.forward(data["obs2"])
                q1_value_next_targ = agent_targ.q1.forward(data["obs2"], next_pi)
                q2_value_next_targ = agent_targ.q2.forward(data["obs2"], next_pi)
                # Entropy-regularized target Q value at next time step
                min_q_value_next_targ = (
                    torch.min(q1_value_next_targ, q2_value_next_targ)
                    - ent_coeff * next_log_pi
                )
                # Target Q value at current time step (from Bellman equation)
                q_value_targ = data["rew"].flatten() + (
                    1 - data["done"].flatten()
                ) * cfg.train.gamma * (min_q_value_next_targ).view(-1)

            q1_value = agent.q1.forward(data["obs"], data["act"]).view(-1)
            q2_value = agent.q2.forward(data["obs"], data["act"]).view(-1)
            # Q network loss functions
            q1_loss = torch.nn.functional.mse_loss(q1_value, q_value_targ)
            q2_loss = torch.nn.functional.mse_loss(q2_value, q_value_targ)
            q_loss = q1_loss + q2_loss

            # Update Q networks
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Policy update
            if global_step % cfg.train.policy_freq == 0:  # delayed update from TD3

                # Freeze Q networks so you don't waste computational effort
                # computing gradients for them during the policy update
                for p in q_params:
                    p.requires_grad = False

                for _ in range(cfg.train.policy_freq):  # compensate for delay
                    pi, log_pi = agent.pi.forward(data["obs"])
                    q1_value_pi = agent.q1.forward(data["obs"], pi)
                    q2_value_pi = agent.q2.forward(data["obs"], pi)
                    min_q_value_pi = torch.min(q1_value_pi, q2_value_pi)
                    # Entropy-regularized policy loss function
                    policy_loss = ((ent_coeff * log_pi) - min_q_value_pi).mean()

                    # Update policy
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    # Automatic entropy tuning
                    if cfg.train.autotune:
                        with torch.no_grad():
                            _, log_pi = agent.pi.forward(data["obs"])
                        # Entropy coefficient loss function
                        ent_coeff_loss = (
                            -log_ent_coeff.exp() * (log_pi + target_entropy)
                        ).mean()

                        # Update entropy coefficient
                        ent_coeff_optimizer.zero_grad()
                        ent_coeff_loss.backward()
                        ent_coeff_optimizer.step()
                        ent_coeff = log_ent_coeff.exp().item()

                # Unfreeze Q networks
                for p in q_params:
                    p.requires_grad = True

            # Target Q networks update
            if global_step % cfg.train.targ_net_freq == 0:
                for param, targ_param in zip(
                    agent.q1.parameters(), agent_targ.q1.parameters()
                ):
                    targ_param.data.copy_(
                        cfg.train.tau * param.data
                        + (1 - cfg.train.tau) * targ_param.data
                    )
                for param, targ_param in zip(
                    agent.q2.parameters(), agent_targ.q2.parameters()
                ):
                    targ_param.data.copy_(
                        cfg.train.tau * param.data
                        + (1 - cfg.train.tau) * targ_param.data
                    )

            # Write training info to tensorboard
            # Note: guard in case there hasn't been a policy update yet
            if (global_step - cfg.train.burn_in) >= cfg.train.policy_freq:
                writer.add_scalar(
                    "losses/q1_value", q1_value.mean().item(), global_step=global_step
                )
                writer.add_scalar(
                    "losses/q2_value", q2_value.mean().item(), global_step=global_step
                )
                writer.add_scalar(
                    "losses/q1_loss", q1_loss.item(), global_step=global_step
                )
                writer.add_scalar(
                    "losses/q2_loss", q2_loss.item(), global_step=global_step
                )
                writer.add_scalar(
                    "losses/policy_loss",
                    policy_loss.item() / 2.0,
                    global_step=global_step,
                )
                writer.add_scalar(
                    "losses/ent_coeff", ent_coeff, global_step=global_step
                )
                # print("SPS : ", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/sps",
                    int(global_step / (time.time() - start_time)),
                    global_step=global_step,
                )
                if cfg.train.autotune:
                    writer.add_scalar(
                        "losses/ent_coeff_loss",
                        ent_coeff_loss.item(),
                        global_step=global_step,
                    )

    # Close envs
    envs.close()

    # Close summary writer
    writer.close()


if __name__ == "__main__":
    main()
