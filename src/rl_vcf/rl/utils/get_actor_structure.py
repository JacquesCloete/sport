from typing import OrderedDict

import numpy as np
from gymnasium.spaces import Space


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
