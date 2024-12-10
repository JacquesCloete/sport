from typing import OrderedDict

import gymnasium as gym
import numpy as np
import safety_gymnasium
from gymnasium.spaces import Box, Space
from numpy.typing import NDArray

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


def process_info(info: dict) -> tuple[NDArray, NDArray]:
    """Process the info dictionary to extract goal_achieved and constraint_violated flags."""
    # Get num_envs
    if "final_info" in info:
        num_envs = np.prod(info["final_info"].shape)
    else:
        num_envs = np.prod(info["goal_met"].shape)
    # Instantiate flags
    goal_achieved = np.full(num_envs, False, dtype=bool)
    constraint_violated = np.full(num_envs, False, dtype=bool)
    # Check goal_met
    if "goal_met" in info:
        goal_achieved[info["_goal_met"]] = info["goal_met"][info["_goal_met"]]
    # Check constraint_violated
    if "constraint_violated" in info:
        constraint_violated[info["_constraint_violated"]] = info["constraint_violated"][
            info["_constraint_violated"]
        ]
    # Check final_info's goal_met and constraint_violated
    if "final_info" in info:
        for env_idx, env_info in enumerate(info["final_info"]):
            if env_info is not None:
                goal_achieved[env_idx] = env_info["goal_met"]
                constraint_violated[env_idx] = env_info["constraint_violated"]

    return goal_achieved, constraint_violated


def get_actor_structure(
    actor_state_dict: OrderedDict,
    obs_space: Space | None = None,
    act_space: Space | None = None,
) -> tuple[list[int], str | None]:
    """
    Get the hidden layers and activation function class name of an actor model from its state dict.

    Optionally, check compatibility against the environment observation space and/or action space.
    """
    if "activation" in actor_state_dict:
        activation = "".join(map(chr, actor_state_dict["activation"].tolist()))
    else:
        activation = None

    hidden_sizes = []
    # TODO: improve extraction of hidden sizes
    for item in actor_state_dict:
        if "net" in item and "weight" in item:
            hidden_sizes.append(actor_state_dict[item].shape[0])
    assert (
        hidden_sizes
    ), "No hidden layers found in the model; make sure your model has mlp variable name 'net'."

    if obs_space is not None:
        input_shape = actor_state_dict["net.0.weight"].shape[1]
        assert (
            input_shape == np.array(obs_space.shape).prod()
        ), f"{np.array(obs_space.shape).prod()} (env obs space) != {input_shape} (actor input shape)"
    if act_space is not None:
        output_shape = actor_state_dict["mu_layer.weight"].shape[0]
        assert output_shape == np.prod(
            act_space.shape
        ), f"{np.prod(act_space.shape)} (env act space) != {output_shape} (actor output shape)"

    return hidden_sizes, activation
