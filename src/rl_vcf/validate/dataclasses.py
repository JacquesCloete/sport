from dataclasses import dataclass

# Structured configs for type checking


@dataclass
class ValidateCommonConfig:
    gym_id: str  # name of gym environment
    seed: int  # rng seed
    total_eps: int  # total no. episodes (scenarios)
    torch_deterministic: bool  # use deterministic torch algs
    cuda: bool  # use gpu
    capture_video: bool  # capture videos of agent over an episode
    capture_video_ep_interval: int  # video capture episode interval
    camera_name: str  # name of camera to use for video capture
    num_envs: int  # no. parallel environments
    clip_action: bool  # clip actions for continuous action spaces
    normalize_observation: bool  # normalize observations for continuous action spaces
    normalize_reward: bool  # normalize rewards for continuous action spaces
    save_db: bool  # save scenario database
    save_db_ep_interval: int  # save scenario database episode interval
    policy_path: str  # relative path to policy to validate
    load_db: bool  # load existing scenario database
    load_db_path: str  # relative path to existing scenario database
    control_rng: bool  # control rng for reproducibility
    env_seed: int | None  # fixed environment seed
