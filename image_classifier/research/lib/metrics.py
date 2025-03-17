from dataclasses import dataclass

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
