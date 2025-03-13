from dataclasses import dataclass
from typing import Literal, Optional

from image_classifier.utils.default_device import default_device


@dataclass
class TrainingConfig:
    epochs_count: int

    learning_rate: float = 0.01
    momentum: float = 0
    weight_decay: float = 0

    num_workers: int = 0
    batch_size: int = 16

    save_model_results_on_exit: bool = False
    save_training_checkpoint_on_exit: bool = False
    training_checkpoint_to_load_path: Optional[str] = None


@dataclass
class ImageClassifierConfig:
    seed: Optional[int]

    training: TrainingConfig

    device: Literal["cuda", "cpu"] = default_device

    model_results_file_path: str = "./image_classifier/research/models-results.json"

    saved_models_directory_path: str = "./bin"


image_classifier_config = ImageClassifierConfig(
    seed=43,
    device="cuda",
    training=TrainingConfig(
        epochs_count=220,
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=128,
        num_workers=7,
        save_training_checkpoint_on_exit=True,
    ),
)
