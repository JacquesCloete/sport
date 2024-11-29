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
