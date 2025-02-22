from dataclasses import dataclass
from typing import Literal, Optional

from image_classifier.utils.default_device import default_device


@dataclass
class TrainingConfig:
    epochs_count: int
    learning_rate: float = 0.01
    num_workers: int = 0
    batch_size: int = 16


@dataclass
class ImageClassifierConfig:
    seed: Optional[int]

    training: TrainingConfig

    device: Literal["cuda", "cpu"] = default_device

    model_results_file_path: str = "./image_classifier/research/models-results.json"


image_classifier_config = ImageClassifierConfig(
    seed=43,
    device="cuda",
    training=TrainingConfig(
        epochs_count=36, learning_rate=0.001, batch_size=64, num_workers=4
    ),
)
