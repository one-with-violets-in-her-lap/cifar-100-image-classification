from dataclasses import asdict, dataclass
import json
import os
from typing import TypeVar

from image_classifier.utils.dataclass_from_dict import (
    create_dataclass_instance_from_dict,
)
from image_classifier.utils.train_test_split import DatasetSplit, TrainTestValue


@dataclass
class NeuralNetMetrics:
    neural_net_name: str

    loss_records_for_each_epoch: list[float]
    """Average model loss records added each epoch"""

    accuracy_records_for_each_epoch: list[float]
    """Average model accuracy records added each epoch

    They're in **percentage** format, e.g. `50`
    """

    def get_best_loss(self):
        return min(self.loss_records_for_each_epoch)

    def get_best_accuracy(self):
        return max(self.accuracy_records_for_each_epoch)

    def __str__(self):
        return (
            f"Best loss - {round(self.get_best_loss(), 2)}, "
            + f"Best accuracy - {round(self.get_best_accuracy(), 2)}%"
        )


@dataclass
class NeuralNetTrainTestMetrics:
    neural_net_name: str

    loss_records_for_each_epoch: TrainTestValue[list[float]]
    """Average model loss records added each epoch"""

    accuracy_records_for_each_epoch: TrainTestValue[list[float]]
    """Average model accuracy records added each epoch

    They're in **percentage** format, e.g. `50`
    """

    def get_best_loss(self, dataset_split: DatasetSplit):
        return min(getattr(self.loss_records_for_each_epoch, dataset_split.value))

    def get_best_accuracy(self, dataset_split: DatasetSplit):
        return max(getattr(self.accuracy_records_for_each_epoch, dataset_split.value))

    def __str__(self):
        best_train_loss = round(self.get_best_loss(DatasetSplit.TRAIN), 2)
        best_test_loss = round(self.get_best_loss(DatasetSplit.TEST), 2)
        best_train_accuracy = round(self.get_best_accuracy(DatasetSplit.TRAIN), 2)
        best_test_accuracy = round(self.get_best_accuracy(DatasetSplit.TEST), 2)
        return (
            f"Best loss - train: {best_train_loss}, test: {best_test_loss} | "
            + f"Best accuracy - train: {best_train_accuracy}%, test: {best_test_accuracy}%"
        )


ModelsResultsT = TypeVar("ModelsResultsT", NeuralNetMetrics, NeuralNetTrainTestMetrics)


def load_models_results(
    file_path: str,
    models_results_type: type[ModelsResultsT],
):
    with open(
        file_path,
        "rt",
        encoding="utf-8",
    ) as models_results_json_stream:
        models_results_dicts: list[dict] = json.load(models_results_json_stream)
        models_results = [
            create_dataclass_instance_from_dict(models_results_type, model_results)
            for model_results in models_results_dicts
        ]

        return models_results


def save_model_results(
    results: ModelsResultsT,
    file_path: str,
):
    models_results_dicts: list[dict] = []

    if os.path.exists(file_path):
        with open(
            file_path,
            "rt",
            encoding="utf-8",
        ) as models_results_json_stream:
            models_results_dicts = json.load(models_results_json_stream)

            models_results_dicts = [
                model_results
                for model_results in models_results_dicts
                if model_results["neural_net_name"] != results.neural_net_name
            ]

    models_results_dicts.append(asdict(results))

    with open(
        file_path,
        "wt",
        encoding="utf-8",
    ) as models_results_json_write_stream:
        models_results_json_write_stream.write(json.dumps(models_results_dicts))

        return models_results_dicts
