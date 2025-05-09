from dataclasses import dataclass, field

from sport.rl.dataclasses import (
    NetworkConfig,
    SafetyTrainCommonConfig,
    TrainCommonConfig,
    WandBConfig,
)
from sport.validate.dataclasses import ValidateCommonConfig

# Structured configs for type checking


@dataclass
class TrainSACConfig:
    policy_lr: float  # policy learning rate
    q_lr: float  # q-network learning rate
    q_weight_decay: float  # q-network weight decay
    adam_epsilon: float  # adam optimizer epsilon
    gamma: float  # discount factor
    buffer_size: int  # replay buffer size
    batch_size: int  # batch size of samples from replay buffer
    burn_in: int  # burn-in before learning starts
    burn_in_train_critic: bool  # whether to train critic during burn-in
    ent_coeff: float  # entropy regularization coefficient
    autotune: bool  # autotune entropy regularization coefficient
    policy_freq: int  # policy update frequency
    targ_net_freq: int  # target network update frequency
    tau: float  # target smoothing polyak coefficient
    targ_ent_coeff: float  # target entropy (scaling coefficient from default)
    per: bool  # use prioritized experience replay
    per_alpha: float  # PER alpha parameter
    per_beta_start: float  # PER beta parameter (start of annealing)
    per_beta_end: float  # PER beta parameter (end of annealing)
    curriculum: bool  # use curriculum learning
    curriculum_levels: list[int]  # curriculum levels
    curriculum_thresholds: list[float]  # curriculum success rate thresholds
    curriculum_window: int  # curriculum success rate checking window size
    expectile_loss: bool  # use expectile loss
    expectile: float  # expectile parameter


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
    safety_common: SafetyTrainCommonConfig = field(
        default_factory=SafetyTrainCommonConfig
    )
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class SACValidateConfig:
    validate_common: ValidateCommonConfig = field(default_factory=ValidateCommonConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
