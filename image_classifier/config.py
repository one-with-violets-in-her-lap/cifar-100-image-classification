from dataclasses import dataclass
from typing import Literal, Optional

from image_classifier.utils.default_device import default_device


@dataclass
class ImageClassifierConfig:
    seed: Optional[int]
    device: Literal["cuda", "cpu"] = default_device


image_classifier_config = ImageClassifierConfig(seed=43, device="cuda")
