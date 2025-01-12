from dataclasses import dataclass, field

from rl_vcf.rl.dataclasses import NetworkConfig, TrainCommonConfig, WandBConfig
from rl_vcf.validate.dataclasses import ValidateCommonConfig

# Structured configs for type checking


@dataclass
class TrainProjectedPPOConfig:
    lr: float  # learning rate
    v_lr: float  # value function learning rate during warmup
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
    base_policy_path: str  # relative path to base policy
    use_base_policy_critic_structure: bool  # use base policy critic network structure
    warmup_total_timesteps: int  # total no. warm-up steps for critic retargeting
    warmup_seed: int  # seed for warm-up
    alpha: float  # Maximum policy ratio for projection
    record_warmup_predicted_discounted_return: (
        bool  # log predicted discounted return during warm-up
    )
    check_max_policy_ratios: bool  # check max policy ratios during training
    check_ref_task_policy: (
        bool  # check reference task policy mean + std during training
    )
    ref_task_policy_path: str  # relative path to reference task policy
    save_db: bool  # save policy projection database
    save_db_ep_interval: int  # interval for saving policy projection database


@dataclass
class ProjectedPPOConfig:
    train: TrainProjectedPPOConfig = field(default_factory=TrainProjectedPPOConfig)
    train_common: TrainCommonConfig = field(default_factory=TrainCommonConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class ProjectedPPOValidateConfig:
    task_policy_path: str  # relative path to task policy
    alpha: float  # Maximum policy ratio for projection
    check_ref_task_policy: (
        bool  # check reference task policy mean + std during validation
    )
    ref_task_policy_path: str  # relative path to reference task policy
    check_max_policy_ratios: bool  # check max policy ratios during validation
    load_policy_projection_db: bool  # load policy projection database
    load_policy_projection_db_path: str  # relative path to policy projection database
    use_alpha_task: bool  # use task policy trained while constrained to alpha
    validate_common: ValidateCommonConfig = field(default_factory=ValidateCommonConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
