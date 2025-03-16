from dataclasses import dataclass
from typing import Literal, Optional

from image_classifier.utils.default_device import default_device


@dataclass
class TrainingConfig:
    epochs_count: int

    learning_rate: float = 0.01
    momentum: float = 0
    weight_decay: float = 0

    save_model_results_on_exit: bool = False
    save_training_checkpoint_on_exit: bool = False


@dataclass
class ImageClassifierConfig:
    seed: Optional[int]

    training: TrainingConfig

    batch_size: int = 16
    num_workers: int = 0

    device: Literal["cuda", "cpu"] = default_device

    model_results_file_path: str = "./image_classifier/research/models-results.json"

    saved_models_directory_path: str = "./bin"

    training_checkpoint_path: Optional[str] = None


# TODO: move to .env/yaml/cmd args
image_classifier_config = ImageClassifierConfig(
    seed=43,
    device="cuda",
    training=TrainingConfig(
        epochs_count=130,
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=0.0005,
        save_training_checkpoint_on_exit=True,
        save_model_results_on_exit=True,
    ),
    batch_size=128,
    num_workers=4,
    training_checkpoint_path="./bin/ResNet base model (training checkpoint) 100 training acc.pth",
)
