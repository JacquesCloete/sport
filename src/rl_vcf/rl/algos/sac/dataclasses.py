from dataclasses import dataclass, field

from rl_vcf.rl.dataclasses import NetworkConfig, TrainCommonConfig, WandBConfig

# Structured configs for type checking


@dataclass
class TrainSACConfig:
    policy_lr: float  # policy learning rate
    q_lr: float  # q-network learning rate
    adam_epsilon: float  # adam optimizer epsilon
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
    train: TrainSACConfig = field(default_factory=TrainSACConfig)
    train_common: TrainCommonConfig = field(default_factory=TrainCommonConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class SACSafetyConfig:
    train: TrainSACConfig = field(default_factory=TrainSACConfig)
    train_common: TrainCommonConfig = field(default_factory=TrainCommonConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    learn_safety: bool = False  # learn to avoid constraint violations
