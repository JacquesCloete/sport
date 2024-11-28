import gymnasium as gym
from gymnasium.spaces import Box


def make_env(
    gym_id: str,
    idx: int,
    seed: int,
    capture_video: bool = False,
    video_episode_interval: int = 100,
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
            env = gym.wrappers.ClipAction(env)  # tanh squashing works better than this
            env = gym.wrappers.NormalizeObservation(env)  # can help performance a lot!
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
