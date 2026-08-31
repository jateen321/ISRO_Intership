from dataclasses import dataclass


DAMAGE_CLASS_NAMES = (
    "background",
    "undamaged",
    "minor",
    "major",
    "destroyed",
)
IGNORE_INDEX = 255
NUM_CLASSES = len(DAMAGE_CLASS_NAMES)


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 512
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    seed: int = 42
