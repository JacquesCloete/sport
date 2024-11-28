from dataclasses import dataclass, field

from rl_vcf.rl.utils.dataclasses import WandBConfig

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
    video_ep_interval: int
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
    hidden_sizes: tuple[int]
    activation: str


@dataclass
class PPOConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
