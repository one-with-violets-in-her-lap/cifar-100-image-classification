from dataclasses import dataclass
from typing import Literal, Optional

from image_classifier.utils.default_device import default_device


@dataclass
class TrainingConfig:
    epochs_count: int
    learning_rate: float = 0.01


@dataclass
class ImageClassifierConfig:
    seed: Optional[int]

    training: TrainingConfig

    batch_size: int = 16
    device: Literal["cuda", "cpu"] = default_device


image_classifier_config = ImageClassifierConfig(
    seed=43,
    batch_size=64,
    device="cuda",
    training=TrainingConfig(epochs_count=50, learning_rate=0.001),
)
