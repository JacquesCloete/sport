import logging
import os
import random
import time

import gymnasium as gym  # noqa: F401
import hydra
import numpy as np
import safety_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_vcf.rl.algos.projected_ppo.core import MLPProjectedActorCritic
from rl_vcf.rl.algos.projected_ppo.dataclasses import ProjectedPPOConfig
from rl_vcf.rl.utils import get_actor_structure, make_env_safety
from rl_vcf.validate.utils import (
    PolicyProjectionDatabase,
    save_policy_projection_database,
)


# Need to run everything inside hydra main function
@hydra.main(
    config_path="config", config_name="projected_ppo_finetune", version_base="1.3"
)
def main(cfg: ProjectedPPOConfig) -> None:
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
    random.seed(cfg.train.warmup_seed)
    np.random.seed(cfg.train.warmup_seed)
    torch.manual_seed(cfg.train.warmup_seed)
    torch.backends.cudnn.deterministic = cfg.train_common.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.train_common.cuda else "cpu"
    )
    if cfg.train_common.env_seed == "None":  # null becomes "None" str in wandb sweep
        env_seed = None
    elif cfg.train_common.env_seed is not None:
        assert isinstance(cfg.train_common.env_seed, int), "env_seed must be type int"
        env_seed = cfg.train_common.env_seed
    # Environment setup for critic retargeting
    # Note: vectorized envs
    envs = safety_gymnasium.vector.SafetySyncVectorEnv(
        [
            make_env_safety(
                cfg.train_common.gym_id,
                i,
                cfg.train.warmup_seed + i,
                False,
                cfg.train_common.capture_video_ep_interval,
                cfg.train_common.clip_action,
                cfg.train_common.normalize_observation,
                cfg.train_common.normalize_reward,
                env_seed=env_seed,
                camera_name=cfg.train_common.camera_name,
            )
            for i in range(cfg.train_common.num_envs)
        ]
    )

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

    # Instantiate policy projection database
    policy_projection_db = PolicyProjectionDatabase(
        alpha=cfg.train.alpha,
        check_ref_task_policy=cfg.train.check_ref_task_policy,
    )

    # Load policy state dict
    abs_base_policy_path = os.path.join(original_dir, cfg.train.base_policy_path)
    assert os.path.exists(
        abs_base_policy_path
    ), "Base policy path {path} does not exist".format(path=abs_base_policy_path)
    loaded_state_dict = torch.load(
        abs_base_policy_path, weights_only=True, map_location=device
    )

    if cfg.train.check_ref_task_policy:
        abs_ref_task_policy_path = os.path.join(
            original_dir, cfg.train.ref_task_policy_path
        )
        assert os.path.exists(
            abs_ref_task_policy_path
        ), "Task policy path {path} does not exist".format(
            path=abs_ref_task_policy_path
        )
        ref_task_loaded_state_dict = torch.load(
            abs_ref_task_policy_path, weights_only=True, map_location=device
        )

    # Construct agent from state dict
    loaded_hidden_sizes, loaded_activation = get_actor_structure(
        loaded_state_dict, envs.single_observation_space, envs.single_action_space
    )

    if cfg.train.use_base_policy_critic_structure:
        critic_hidden_sizes = loaded_hidden_sizes
        critic_activation = loaded_activation
    else:
        critic_hidden_sizes = cfg.network.hidden_sizes
        critic_activation = cfg.network.activation

    if cfg.train.check_ref_task_policy:
        ref_task_loaded_hidden_sizes, ref_task_loaded_activation = get_actor_structure(
            ref_task_loaded_state_dict,
            envs.single_observation_space,
            envs.single_action_space,
        )
    else:
        ref_task_loaded_hidden_sizes = loaded_hidden_sizes
        ref_task_loaded_activation = loaded_activation

    # Create agent
    agent = MLPProjectedActorCritic(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        hidden_sizes_base=loaded_hidden_sizes,
        activation_base=eval("nn." + loaded_activation + "()"),
        hidden_sizes_task=loaded_hidden_sizes,
        activation_task=eval("nn." + loaded_activation + "()"),
        hidden_sizes_critic=critic_hidden_sizes,
        activation_critic=eval("nn." + critic_activation + "()"),
        alpha=cfg.train.alpha,
        check_max_policy_ratios=cfg.train.check_max_policy_ratios,
        check_ref_task_policy=cfg.train.check_ref_task_policy,
        hidden_sizes_ref_task=ref_task_loaded_hidden_sizes,
        activation_ref_task=eval("nn." + ref_task_loaded_activation + "()"),
    ).to(device)
    print(agent)

    agent.pi_base.load_state_dict(loaded_state_dict, strict=True)
    agent.pi_task.load_state_dict(loaded_state_dict, strict=True)
    if cfg.train.check_ref_task_policy:
        agent.pi_ref_task.load_state_dict(ref_task_loaded_state_dict, strict=True)
    agent.to(device)

    # Create optimizers
    optimizer = optim.Adam(
        agent.parameters(), lr=cfg.train.lr, eps=cfg.train.adam_epsilon
    )  # default torch epsilon of 1e-8 found to be better
    # Optimizer for critic retargetting
    v_optimizer = optim.Adam(
        agent.v.parameters(), lr=cfg.train.v_lr, eps=cfg.train.adam_epsilon
    )

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
    # Store untransformed actions when using state-dependent std (SAC-like policy)
    x_ts = torch.zeros(
        (cfg.train.num_steps, cfg.train_common.num_envs)
        + envs.single_action_space.shape
    ).to(device)

    ########################################################################################

    # Freeze task policy network
    for p in agent.pi_task.parameters():
        p.requires_grad = False

    # Critic retargetting stage (learn critic for the base policy in the new environment)
    warmup_global_step = 0
    start_time = time.time()

    if env_seed is not None:
        obs, info = envs.reset()  # env seed is already set using the SeedWrapper
    else:
        obs, info = envs.reset(
            seed=[cfg.train.warmup_seed + i for i in range(cfg.train_common.num_envs)]
        )

    next_obs = torch.Tensor(obs).to(device)

    next_done = torch.zeros(cfg.train_common.num_envs).to(device)

    batch_size = int(cfg.train.num_steps * cfg.train_common.num_envs)
    num_updates = cfg.train.warmup_total_timesteps // batch_size
    minibatch_size = int(batch_size // cfg.train.num_minibatches)

    if cfg.train.record_warmup_predicted_discounted_return:
        # Value prediction at start of episode for logging
        pred_return = agent.v.forward(next_obs[0]).item()
        discounted_return = 0
        discount_factor = 1.0

    for update in tqdm(range(1, num_updates + 1)):  # update networks using batch data
        # Annealing the learning rate
        if cfg.train.anneal_lr:
            # annealing helps agents obtain higher episodic return
            frac = 1.0 - (update - 1.0) / num_updates  # linear annealing (to 0)
            lr_now = cfg.train.v_lr * frac
            v_optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, cfg.train.num_steps):  # collect batch data
            # note the batch data gets overwritten in each update loop
            warmup_global_step += 1 * cfg.train_common.num_envs
            observations[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():  # don't need to cache gradients during rollout
                act, logprob, _, value, x_t = agent.get_base_action_and_value(next_obs)
                # Need to also store untransformed actions for state-dependent std
                x_ts[step] = x_t
                values[step] = value.flatten()
            actions[step] = act
            logprobs[step] = logprob

            # Environment step
            next_obs, rew, cost, term, trunc, info = envs.step(act.cpu().numpy())
            rewards[step] = torch.tensor(rew).to(device).view(-1)
            done = np.logical_or(
                term,
                trunc,
            )  # for now, let done be when terminated (task success/failure) or truncated (time limit)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            # Log episodic returns at start of episode to check how critic is doing
            if cfg.train.record_warmup_predicted_discounted_return:
                discounted_return += rew[0] * discount_factor
                discount_factor *= cfg.train.gamma
                if done[0]:
                    writer.add_scalar(
                        "charts/warmup_episodic_discounted_return",
                        discounted_return,
                        global_step=warmup_global_step,
                    )
                    writer.add_scalar(
                        "charts/warmup_predicted_discounted_return",
                        pred_return,
                        global_step=warmup_global_step,
                    )
                    # get predicted return at start of next episode
                    # note: if env_seed is set, this will always be the same until critic is next updated
                    pred_return = agent.v.forward(next_obs[0]).item()
                    discounted_return = 0
                    discount_factor = 1.0

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
        # Use untransformed actions for state-dependent std
        b_actions = x_ts.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update critic network
        b_inds = np.arange(batch_size)
        clipfracs = []  # measures how often clipping is triggered
        for epoch in range(cfg.train.update_epochs):
            np.random.shuffle(b_inds)  # shuffle data in batch
            for start in range(0, batch_size, minibatch_size):
                # update using each minibatch (don't use the entire batch for one update!)
                # fetch minibatches by iterating through batch (don't randomly sample data!)
                end = start + minibatch_size
                mb_inds = b_inds[start:end]  # get minibatch indices for update step

                # Get new values
                _, new_logprobs, entropies, new_values, _ = (
                    agent.get_base_action_and_value(
                        b_observations[mb_inds], b_actions[mb_inds]
                    )
                )

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

                # Update critic network
                v_optimizer.zero_grad()
                v_loss.backward()
                # Note: clip gradient norm for stability
                # Slightly improves performance
                nn.utils.clip_grad_norm_(agent.v.parameters(), cfg.train.max_grad_norm)
                v_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # explained variance tells you if your value function is a good indicator of your returns
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Write training info to tensorboard
        writer.add_scalar(
            "charts/warmup_learning_rate",
            v_optimizer.param_groups[0]["lr"],
            global_step=warmup_global_step,
        )
        writer.add_scalar(
            "losses/warmup_value_loss", v_loss.item(), global_step=warmup_global_step
        )
        writer.add_scalar(
            "losses/warmup_explained_variance",
            explained_var,
            global_step=warmup_global_step,
        )
        writer.add_scalar(
            "charts/warmup_sps",
            int(warmup_global_step / (time.time() - start_time)),
            global_step=warmup_global_step,
        )

    # Unfreeze Q networks
    for p in agent.pi_task.parameters():
        p.requires_grad = True

    ########################################################################################

    # Seeding
    random.seed(cfg.train_common.seed)
    np.random.seed(cfg.train_common.seed)
    torch.manual_seed(cfg.train_common.seed)
    torch.backends.cudnn.deterministic = cfg.train_common.torch_deterministic

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
                cfg.train_common.clip_action,
                cfg.train_common.normalize_observation,
                cfg.train_common.normalize_reward,
                env_seed=env_seed,
                camera_name=cfg.train_common.camera_name,
            )
            for i in range(cfg.train_common.num_envs)
        ]
    )

    # Episode counter
    # To be consistent with episode tracking for videos, track episode of each env separately
    episode_count = np.array([0] * cfg.train_common.num_envs, dtype=int)

    # Start training
    global_step = 0
    start_time = time.time()

    if env_seed is not None:
        obs, info = envs.reset()  # env seed is already set using the SeedWrapper
    else:
        obs, info = envs.reset(
            seed=[cfg.train_common.seed + i for i in range(cfg.train_common.num_envs)]
        )

    next_obs = torch.Tensor(obs).to(device)

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
                act, logprob, _, value, x_t = agent.get_projected_action_and_value(
                    next_obs
                )

                policy_projection_db.update(agent, episode_count)

                if cfg.train.check_max_policy_ratios:
                    for i in range(cfg.train_common.num_envs):
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
                        if cfg.train.check_ref_task_policy:
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
                if cfg.train.check_ref_task_policy:
                    for i in range(cfg.train_common.num_envs):
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
                # Need to also store untransformed actions for state-dependent std
                x_ts[step] = x_t
                values[step] = value.flatten()
            actions[step] = act
            logprobs[step] = logprob

            # Environment step
            next_obs, rew, cost, term, trunc, info = envs.step(act.cpu().numpy())
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

            if done[0]:
                if cfg.train_common.save_model:
                    # To be consistent with episode tracking for videos,
                    # use episode number of env 0 (not total episode count!)
                    if episode_count[0] % cfg.train_common.save_model_ep_interval == 0:
                        # Save model
                        model_name = f"models/rl-model-episode-{episode_count[0]}.pt"
                        torch.save(agent.state_dict(), model_name)
                        # Save policy separately as well
                        policy_name = (
                            f"policies/rl-task-policy-episode-{episode_count[0]}.pt"
                        )
                        torch.save(agent.pi_task.state_dict(), policy_name)

                writer.add_scalar(
                    "charts/episode_count_per_env",
                    episode_count[0],
                    global_step=global_step,
                )
            # Increment episode counter
            episode_count += done

            # Save policy projection database
            if cfg.train.save_db:
                if episode_count[0] % cfg.train.save_db_ep_interval == 0:
                    save_policy_projection_database(
                        policy_projection_db, "policy_projection_db.pkl"
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
        # Use untransformed actions for state-dependent std
        b_actions = x_ts.reshape((-1,) + envs.single_action_space.shape)
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

                # Get policy ratio for clipping and new values
                _, new_logprobs, entropies, new_values, _ = (
                    agent.get_unprojected_action_and_value(
                        b_observations[mb_inds], b_actions[mb_inds]
                    )
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

    if cfg.train_common.save_model:
        # Save trained model
        model_name = "models/rl-model-final.pt"
        torch.save(agent.state_dict(), model_name)
        # Save trained task policy separately as well
        policy_name = "policies/rl-task-policy-final.pt"
        torch.save(agent.pi_task.state_dict(), policy_name)

    # Save final policy projection database
    if cfg.train.save_db:
        save_policy_projection_database(
            policy_projection_db, "policy_projection_db.pkl"
        )

    # Close envs
    envs.close()

    # Close summary writer
    writer.close()


if __name__ == "__main__":
    main()
