from dataclasses import dataclass


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
