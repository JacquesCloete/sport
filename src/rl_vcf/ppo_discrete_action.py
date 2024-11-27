import os
import random
import time
from dataclasses import dataclass, field

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
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def make_env(gym_id: str, seed: int, idx: int, capture_video: bool) -> gym.Env:
    """Create the environment."""

    def thunk():
        env = gym.make(
            gym_id,
            render_mode="rgb_array",  # need to set render mode for video recording
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:  # record only the environment with idx 0
                env = gym.wrappers.RecordVideo(
                    env,
                    "videos",
                    episode_trigger=lambda t: t % 100 == 0,  # every 100 episodes
                )
        # env.seed(seed) # Doesn't work anymore, now set seed using env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0) -> nn.Linear:
    """Initialize layer weights."""
    # Note: different to standard torch initialization methods
    torch.nn.init.orthogonal_(layer.weight, std)  # outperforms default init
    torch.nn.init.constant_(layer.bias, bias_const)  # start with 0 bias
    return layer


class Agent(nn.Module):
    """Agent class."""

    def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
        super(Agent, self).__init__()
        # separate policy and value networks generally lead to better performance
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # output layer uses different std!
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            # std of 0.01 ensures the layer parameters have similar scalar values so probability of picking each action will be similar
            # using np.array(envs.single_action_space.shape).prod() doesn't work here?
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value from critic."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from actor and get value from critic."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# Structured configs for type checking
@dataclass
class TrainConfig:
    gym_id: str
    learning_rate: float
    anneal_lr: bool
    adam_epsilon: float
    seed: int
    total_timesteps: int
    torch_deterministic: bool
    cuda: bool
    capture_video: bool
    num_envs: int
    num_steps: int
    gae: bool
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    norm_adv: bool
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float


@dataclass
class WandBConfig:
    track: bool
    project: int
    entity: str | None
    group: str | None


@dataclass
class PPOConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


# Need to run everything inside hydra main function
@hydra.main(config_path="config", config_name="ppo_discrete_action", version_base="1.3")
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
    wandb.config = OmegaConf.to_container(
        cfg.train, resolve=True, throw_on_missing=True
    )

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
        % ("\n".join([f"|{key}|{value}|" for key, value in cfg.train.items()])),
    )

    # Seeding
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.backends.cudnn.deterministic = cfg.train.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.train.cuda else "cpu"
    )

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                cfg.train.gym_id,
                cfg.train.seed + i,
                i,
                cfg.train.capture_video,
            )
            for i in range(cfg.train.num_envs)
        ]
    )

    # Create agent
    agent = Agent(envs).to(device)
    print(agent)

    optimizer = optim.Adam(
        agent.parameters(), lr=cfg.train.learning_rate, eps=cfg.train.adam_epsilon
    )  # default torch epsilon of 1e-8 found to be better

    # ALGO logic: Storage setup
    observations = torch.zeros(
        (cfg.train.num_steps, cfg.train.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (cfg.train.num_steps, cfg.train.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.train.num_steps, cfg.train.num_envs)).to(device)
    rewards = torch.zeros((cfg.train.num_steps, cfg.train.num_envs)).to(device)
    dones = torch.zeros((cfg.train.num_steps, cfg.train.num_envs)).to(device)
    values = torch.zeros((cfg.train.num_steps, cfg.train.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(
        envs.reset(seed=[cfg.train.seed + i for i in range(cfg.train.num_envs)])[
            0  # observations are first element of env reset output
        ]
    ).to(device)
    next_done = torch.zeros(cfg.train.num_envs).to(device)

    batch_size = int(cfg.train.num_steps * cfg.train.num_envs)
    num_updates = cfg.train.total_timesteps // batch_size
    minibatch_size = int(batch_size // cfg.train.num_minibatches)

    for update in tqdm(range(1, num_updates + 1)):  # update networks using batch data
        # Annealing the learning rate
        if cfg.train.anneal_lr:
            # annealing helps agents obtain higher episodic return
            frac = 1.0 - (update - 1.0) / num_updates  # linear annealing (to 0)
            lr_now = cfg.train.learning_rate * frac
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, cfg.train.num_steps):  # collect batch data
            # note the batch data gets overwritten in each update loop
            global_step += 1 * cfg.train.num_envs
            observations[step] = next_obs
            dones[step] = next_done

            # ALGO logic: action logic
            with torch.no_grad():  # don't need to cache gradients during rollout
                act, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = act
            logprobs[step] = logprob

            # Environment step
            next_obs, rew, term, trunc, info = envs.step(act.cpu().numpy())
            rewards[step] = torch.tensor(rew).to(device).view(-1)
            done = (
                term | trunc
            )  # for now, let done be when terminated (task success/failure) or truncated (time limit)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)
            for _, val in info.items():
                # note info is usually empty but gets populated at end of episode
                # also, the current gymnasium implementation of info seems absolutely terrible?
                # a dict of lists, each list containing a mix of different types, including dicts??
                for item in val:
                    if isinstance(item, dict) and "episode" in item.keys():
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
                        break

        # bootstrap reward if not done (take value at next obs as end-of-roolout value)
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
                    b_observations[mb_inds], b_actions.long()[mb_inds]
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
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    # Close envs
    envs.close()

    # Close summary writer
    writer.close()


if __name__ == "__main__":
    main()
