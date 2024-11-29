from dataclasses import dataclass, field

from rl_vcf.rl.utils.dataclasses import NetworkConfig, WandBConfig

# Structured configs for type checking


@dataclass
class TrainConfig:
    gym_id: str  # name of gym environment
    policy_lr: float  # policy learning rate
    q_lr: float  # q-network learning rate
    adam_epsilon: float  # adam optimizer epsilon
    seed: int  # rng seed
    total_timesteps: int  # total no. environment interactions
    torch_deterministic: bool  # use deterministic torch algs
    cuda: bool  # use gpu
    capture_video: bool  # capture videos of agent over an episode
    video_ep_interval: int  # video capture episode interval
    num_envs: int  # no. parallel environments
    gamma: float  # discount factor
    buffer_size: int  # replay buffer size
    batch_size: int  # batch size of samples from replay buffer
    burn_in: int  # burn-in before learning starts
    ent_coef: float  # entropy regularization coefficient
    autotune: bool  # autotune entropy regularization coefficient
    policy_freq: int  # policy update frequency
    targ_net_freq: int  # target network update frequency
    tau: float  # target smoothing polyak coefficient


@dataclass
class SACConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
