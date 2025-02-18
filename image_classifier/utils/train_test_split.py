from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


TrainTestValueT = TypeVar("TrainTestValueT")


@dataclass
class TrainTestValue(Generic[TrainTestValueT]):
    train: TrainTestValueT
    test: TrainTestValueT
