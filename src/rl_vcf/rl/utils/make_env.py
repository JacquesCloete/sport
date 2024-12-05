import gymnasium as gym
import safety_gymnasium
from gymnasium.spaces import Box

# Gymnasium 0.28.1 (used by safety_gymnasium) breaks RecordVideo
# Use fork JacquesCloete/Gymnasium, branch jacques/v0.28.1, to fix


def make_env(
    gym_id: str,
    idx: int,
    seed: int,
    capture_video: bool = False,
    video_episode_interval: int = 100,
    preprocess_envs: bool = False,
) -> gym.Env:
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
                    episode_trigger=lambda t: t % video_episode_interval == 0,
                )
        # Preprocessing for continuous action spaces
        if isinstance(env.action_space, Box):
            if preprocess_envs:
                env = gym.wrappers.ClipAction(
                    env
                )  # tanh squashing works better than this
                env = gym.wrappers.NormalizeObservation(
                    env
                )  # can help performance a lot!
                # env = gym.wrappers.TransformObservation(
                #     env, lambda obs: np.clip(obs, -10.0, 10.0)
                # )  # observation clipping after normalization doesn't usually help but sometimes can
                env = gym.wrappers.NormalizeReward(env)  # can help performance a lot!
                # env = gym.wrappers.TransformReward(
                #     env, lambda rew: np.clip(rew, -10.0, 10.0)
                # )  # reward clipping after normalization has no evidence of being helpful
                # env.seed(seed) # Doesn't work anymore, now set seed using env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_safety(
    gym_id: str,
    idx: int,
    seed: int,
    capture_video: bool = False,
    video_episode_interval: int = 100,
    preprocess_envs: bool = False,
) -> gym.Env:
    """Create the environment."""

    def thunk():
        # see safety_gymnasium/builder.py to see kwargs
        env = safety_gymnasium.make(
            gym_id,
            render_mode="rgb_array",  # need to set render mode for video recording
            camera_name="fixedfar",
        )
        # wrap to gymnasium
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        # flatten observations if necessary (e.g. for LTL task envs)
        # those envs start unflattened because it makes LTL stuff easier
        # you can also turn on flattening by default by adding the param
        # "observation_flatten": True to the config dict when registering the env
        env = gym.wrappers.FlattenObservation(env)
        # Add all the gymnasium wrappers we want
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:  # record only the environment with idx 0
                env = gym.wrappers.RecordVideo(
                    env,
                    "videos",
                    episode_trigger=lambda t: t % video_episode_interval == 0,
                )
        # Preprocessing for continuous action spaces
        if isinstance(env.action_space, Box):
            if preprocess_envs:
                env = gym.wrappers.ClipAction(
                    env
                )  # tanh squashing works better than this
                env = gym.wrappers.NormalizeObservation(
                    env
                )  # can help performance a lot!
                # env = gym.wrappers.TransformObservation(
                #     env, lambda obs: np.clip(obs, -10.0, 10.0)
                # )  # observation clipping after normalization doesn't usually help but sometimes can
                env = gym.wrappers.NormalizeReward(env)  # can help performance a lot!
                # env = gym.wrappers.TransformReward(
                #     env, lambda rew: np.clip(rew, -10.0, 10.0)
                # )  # reward clipping after normalization has no evidence of being helpful
                # env.seed(seed) # Doesn't work anymore, now set seed using env.reset(seed=seed)
        # wrap back to safety gymnasium
        env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
