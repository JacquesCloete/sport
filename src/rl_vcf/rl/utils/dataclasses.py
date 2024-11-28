from dataclasses import dataclass

# Structured configs for type checking


@dataclass
class WandBConfig:
    track: bool
    project: str | None
    entity: str | None
    group: str | None
