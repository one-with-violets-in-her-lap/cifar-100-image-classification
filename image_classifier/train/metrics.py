from dataclasses import dataclass


@dataclass
class NeuralNetMetrics:
    loss: float
    accuracy: float
    """Model accuracy as a percentage, e.g. `50`"""

    def __str__(self):
        return f"Loss - {round(self.loss, 2)}, Accuracy - {round(self.accuracy, 2)}%"
