from dataclasses import dataclass, field

from rl_vcf.rl.dataclasses import NetworkConfig, TrainCommonConfig, WandBConfig

# Structured configs for type checking


@dataclass
class TrainPPOConfig:
    lr: float  # learning rate
    anneal_lr: bool  # toggle learning rate annealing
    adam_epsilon: float  # adam optimizer epsilon
    num_steps: int  # no. steps per environment per policy rollout
    gae: bool  # use generalized advantage estimation
    gamma: float  # discount factor
    gae_lambda: float  # lambda for GAE
    num_minibatches: int  # no. minibatches per epoch per update step
    update_epochs: int  # no. epochs per update step
    norm_adv: bool  # use advantage normalization
    clip_coef: float  # surrogate clipping coefficient
    clip_vloss: bool  # clip value function loss
    ent_coef: float  # entropy regularization coefficient
    vf_coef: float  # value function loss coefficient
    max_grad_norm: float  # max norm for gradient clipping
    target_kl: float | None  # target KL divergence for early stopping
    state_dependent_std: bool  # use state dependent std for continuous actions


@dataclass
class PPOConfig:
    train: TrainPPOConfig = field(default_factory=TrainPPOConfig)
    train_common: TrainCommonConfig = field(default_factory=TrainCommonConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
