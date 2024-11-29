import os
import random
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_vcf.rl.algos.ppo.core import MLPActorCritic
from rl_vcf.rl.algos.ppo.dataclasses import PPOConfig
from rl_vcf.rl.utils.make_env import make_env


# Need to run everything inside hydra main function
@hydra.main(config_path="config", config_name="ppo", version_base="1.3")
def main(cfg: PPOConfig) -> None:
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
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                cfg.train_common.gym_id,
                i,
                cfg.train_common.seed + i,
                cfg.train_common.capture_video,
                cfg.train_common.video_ep_interval,
                cfg.train_common.preprocess_envs,
            )
            for i in range(cfg.train_common.num_envs)
        ]
    )

    # Create agent
    agent = MLPActorCritic(
        envs.single_observation_space,
        envs.single_action_space,
        cfg.network.hidden_sizes,
        eval("nn." + cfg.network.activation + "()"),
    ).to(device)
    print(agent)

    # Create optimizer
    optimizer = optim.Adam(
        agent.parameters(), lr=cfg.train.lr, eps=cfg.train.adam_epsilon
    )  # default torch epsilon of 1e-8 found to be better

    # ALGO LOGIC: Storage setup
    observations = torch.zeros(
        (cfg.train.num_steps, cfg.train_common.num_envs)
        + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (cfg.train.num_steps, cfg.train_common.num_envs)
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.train.num_steps, cfg.train_common.num_envs)).to(device)
    rewards = torch.zeros((cfg.train.num_steps, cfg.train_common.num_envs)).to(device)
    dones = torch.zeros((cfg.train.num_steps, cfg.train_common.num_envs)).to(device)
    values = torch.zeros((cfg.train.num_steps, cfg.train_common.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(
        envs.reset(
            seed=[cfg.train_common.seed + i for i in range(cfg.train_common.num_envs)]
        )[
            0  # observations are first element of env reset output
        ]
    ).to(device)
    next_done = torch.zeros(cfg.train_common.num_envs).to(device)

    batch_size = int(cfg.train.num_steps * cfg.train_common.num_envs)
    num_updates = cfg.train_common.total_timesteps // batch_size
    minibatch_size = int(batch_size // cfg.train.num_minibatches)

    for update in tqdm(range(1, num_updates + 1)):  # update networks using batch data
        # Annealing the learning rate
        if cfg.train.anneal_lr:
            # annealing helps agents obtain higher episodic return
            frac = 1.0 - (update - 1.0) / num_updates  # linear annealing (to 0)
            lr_now = cfg.train.lr * frac
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, cfg.train.num_steps):  # collect batch data
            # note the batch data gets overwritten in each update loop
            global_step += 1 * cfg.train_common.num_envs
            observations[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():  # don't need to cache gradients during rollout
                act, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = act
            logprobs[step] = logprob

            # Environment step
            next_obs, rew, term, trunc, info = envs.step(act.cpu().numpy())
            rewards[step] = torch.tensor(rew).to(device).view(-1)
            done = np.logical_or(
                term,
                trunc,
            )  # for now, let done be when terminated (task success/failure) or truncated (time limit)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)
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

        # Bootstrap reward if not done (take value at next obs as end-of-rollout value)
        # Note that for vectorized environments, each environment auto-resets when done
        # So when we step, we actually see the initial observation of the next episode
        # So if we want the final observation of the previous episode, we look in info
        # But we don't have to do that here because we only bootstrap if not done!
        # See SAC implementation for when you need the final observation when done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if cfg.train.gae:
                # GAE, implemented as in the original PPO code (causes slightly different adv/ret!)
                # performs better than N-step returns
                advantages = torch.zeros_like(rewards).to(device)
                last_gae_lam = 0
                for t in reversed(range(cfg.train.num_steps)):
                    if t == cfg.train.num_steps - 1:
                        # bootstrap if not done
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = (
                        rewards[t]
                        + cfg.train.gamma * next_values * next_non_terminal
                        - values[t]
                    )
                    advantages[t] = last_gae_lam = (
                        delta
                        + cfg.train.gamma
                        * cfg.train.gae_lambda
                        * next_non_terminal
                        * last_gae_lam
                    )
                returns = (
                    advantages + values
                )  # corresponds to TD(lambda) for value estimation
            else:
                # the typical method of advantage calculation
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(cfg.train.num_steps)):
                    if t == cfg.train.num_steps - 1:
                        # bootstrap if not done
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = (
                        rewards[t] + cfg.train.gamma * next_non_terminal * next_return
                    )
                advantages = returns - values

        # flatten the batch
        b_observations = observations.reshape(
            (-1,) + envs.single_observation_space.shape
        )
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update actor and critic networks
        b_inds = np.arange(batch_size)
        clipfracs = []  # measures how often clipping is triggered
        for epoch in range(cfg.train.update_epochs):
            np.random.shuffle(b_inds)  # shuffle data in batch
            for start in range(0, batch_size, minibatch_size):
                # update using each minibatch (don't use the entire batch for one update!)
                # fetch minibatches by iterating through batch (don't randomly sample data!)
                end = start + minibatch_size
                mb_inds = b_inds[start:end]  # get minibatch indices for update step

                # Get policy ratio for clipping
                _, new_logprobs, entropies, new_values = agent.get_action_and_value(
                    b_observations[mb_inds], b_actions[mb_inds]
                )
                logratios = new_logprobs - b_logprobs[mb_inds]
                ratios = logratios.exp()

                with torch.no_grad():
                    # kl helps understand the aggressiveness of each update
                    # calculate approx_kl: http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratios).mean()
                    # the below implementation is unbiased and has less variance
                    approx_kl = ((ratios - 1.0) - logratios).mean()
                    clipfracs += [
                        ((ratios - 1.0).abs() > cfg.train.clip_coef)
                        .float()
                        .mean()
                        .cpu()
                        .numpy()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.train.norm_adv:
                    # Normalize advantages, at minibatch level
                    # Doesn't help much
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Clipped actor loss
                pg_loss1 = -mb_advantages * ratios
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratios, 1 - cfg.train.clip_coef, 1 + cfg.train.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss
                # Note: no evidence of clipping helping with performance
                # In fact, it can even hurt performance!
                new_values = new_values.view(-1)
                if cfg.train.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_values - b_values[mb_inds],
                        -cfg.train.clip_coef,
                        cfg.train.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()

                # Original PPO also includes entropy regularization to improve exploration
                # Note: no evidence of entropy regularization helping with performance
                # for continuous control environments
                # But we might find it helpful for encouraging an exploratory policy
                entropy_loss = entropies.mean()

                # Combined loss
                loss = (
                    pg_loss
                    + v_loss * cfg.train.vf_coef
                    - entropy_loss * cfg.train.ent_coef
                )

                # Update networks
                optimizer.zero_grad()
                loss.backward()
                # Note: clip gradient norm for stability
                # Slightly improves performance
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.train.max_grad_norm)
                optimizer.step()

            # Stop the policy update for the batch if the KL divergence has grown too large so as
            # to exceed a target KL divergence threshold
            # (could alternatively implement at the minibatch level)
            # spinningup uses 0.015

            if cfg.train.target_kl == "None":  # null becomes "None" str in wandb sweep
                pass
            elif cfg.train.target_kl is not None:
                if approx_kl > cfg.train.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # explained variance tells you if your value function is a good indicator of your returns
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Write training info to tensorboard
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS : ", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/sps", int(global_step / (time.time() - start_time)), global_step
        )

    # Close envs
    envs.close()

    # Close summary writer
    writer.close()


if __name__ == "__main__":
    main()
