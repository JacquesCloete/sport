from dataclasses import dataclass

# Structured configs for type checking


@dataclass
class NetworkConfig:
    hidden_sizes: tuple[int]  # hidden layer sizes
    activation: str  # activation function


@dataclass
class WandBConfig:
    track: bool  # track experiment with wandb
    project: str | None  # wandb project name
    entity: str | None  # wandb entity (team) name
    group: str | None  # wandb experiment group name


@dataclass
class TrainCommonConfig:
    gym_id: str  # name of gym environment
    seed: int  # rng seed
    total_timesteps: int  # total no. environment interactions
    torch_deterministic: bool  # use deterministic torch algs
    cuda: bool  # use gpu
    capture_video: bool  # capture videos of agent over an episode
    capture_video_ep_interval: int  # video capture episode interval
    num_envs: int  # no. parallel environments
    clip_action: bool  # clip actions for continuous action spaces
    normalize_observation: bool  # normalize observations for continuous action spaces
    normalize_reward: bool  # normalize rewards for continuous action spaces
    save_model: bool  # save model weights
    save_model_ep_interval: int  # save model weights episode interval


@dataclass
class SafetyTrainCommonConfig:
    learn_safety: bool  # learn to avoid constraint violations
    sparse_reward: float  # sparse reward for achieving the goal
    sparse_penalty: float  # sparse penalty for violating a constraint
    dense_reward_coeff: float  # reward coefficient for getting closer to the goal
    dense_penalty_coeff: float  # penalty coefficient for getting closer to a constraint
    inactivity_penalty_coeff: float  # penalty coefficient for not doing anything
