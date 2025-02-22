from typing import Callable
from torch import nn
import torch

from image_classifier.train.metrics import NeuralNetMetrics


def test_neural_net(
    neural_net: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    accuracy_function: Callable,
    device: torch.types.Device,
):
    neural_net.eval()

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0
    total_accuracy = 0

    with torch.inference_mode():
        for batch in test_dataloader:
            images: torch.Tensor = batch[0].to(device=device)
            ideal_classes_output: torch.Tensor = batch[1].to(device=device)

            raw_output = neural_net(images)

            batch_loss = loss_function(raw_output, ideal_classes_output)
            total_loss += batch_loss.item()

            predicted_probabilities: torch.Tensor = raw_output.softmax(0)
            predicted_classes = predicted_probabilities.argmax(dim=1)

            batch_accuracy = accuracy_function(predicted_classes, ideal_classes_output)
            total_accuracy += batch_accuracy.item()

        batches_count = len(test_dataloader)
        average_loss = total_loss / batches_count
        average_accuracy = (total_accuracy / batches_count) * 100

        return NeuralNetMetrics(loss=average_loss, accuracy=average_accuracy)
