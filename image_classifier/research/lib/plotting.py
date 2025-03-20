from matplotlib import pyplot as plt

from image_classifier.research.lib.metrics import (
    NeuralNetTrainTestMetrics,
)
from image_classifier.utils.train_test_split import DatasetSplit


DATASET_SPLIT_LINE_CHART_COLORS: dict[DatasetSplit, str] = {
    DatasetSplit.TRAIN: "blue",
    DatasetSplit.TEST: "red",
}


def plot_loss_and_accuracy(*metrics_items: NeuralNetTrainTestMetrics):
    figure = plt.figure(figsize=(16, 9))

    current_subplot_index = 0

    for metrics in metrics_items:
        # Loss curve
        current_subplot_index += 1
        figure.add_subplot(len(metrics_items), 2, current_subplot_index)
        plt.title(f"{metrics.neural_net_name} | Loss")

        plt.plot(
            metrics.loss_records_for_each_epoch.test,
            c=DATASET_SPLIT_LINE_CHART_COLORS[DatasetSplit.TEST],
            label=DatasetSplit.TEST.value,
        )
        plt.plot(
            metrics.loss_records_for_each_epoch.train,
            c=DATASET_SPLIT_LINE_CHART_COLORS[DatasetSplit.TRAIN],
            label=DatasetSplit.TRAIN.value,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Accuracy curve
        current_subplot_index += 1
        figure.add_subplot(len(metrics_items), 2, current_subplot_index)
        plt.title(f"{metrics.neural_net_name} | Accuracy")

        plt.plot(
            metrics.accuracy_records_for_each_epoch.test,
            c=DATASET_SPLIT_LINE_CHART_COLORS[DatasetSplit.TEST],
            label=DatasetSplit.TEST.value,
        )
        plt.plot(
            metrics.accuracy_records_for_each_epoch.train,
            c=DATASET_SPLIT_LINE_CHART_COLORS[DatasetSplit.TRAIN],
            label=DatasetSplit.TRAIN.value,
        )


        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

    plt.tight_layout(w_pad=5)

    plt.legend()
    plt.show()
