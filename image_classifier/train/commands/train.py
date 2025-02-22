from typing import Callable
import click
import torch
from torch import nn
from torchmetrics import Accuracy

from image_classifier.data.cifar_100 import (
    create_cifar_100_dataloaders,
    cifar_100_train_dataset,
)
from image_classifier.models.custom_convolutional_net import CustomConvolutionalNet
from image_classifier.config import image_classifier_config
from image_classifier.train.metrics import NeuralNetMetrics
from image_classifier.train.test import test_neural_net


@click.command()
def train():
    dataloaders = create_cifar_100_dataloaders(
        image_classifier_config.batch_size, image_classifier_config.num_workers
    )
    train_dataloader = dataloaders.train
    test_dataloader = dataloaders.test

    neural_net = CustomConvolutionalNet(classes_count=len(cifar_100_train_dataset.classes))
    neural_net.to(device=image_classifier_config.device)

    optimizer = torch.optim.Adam(
        neural_net.parameters(), lr=image_classifier_config.training.learning_rate
    )

    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device=image_classifier_config.device)

    accuracy_function = Accuracy(
        "multiclass", num_classes=len(cifar_100_train_dataset.classes)
    )
    accuracy_function.to(device=image_classifier_config.device)

    click.echo(
        f"Starting training loop. Train dataset size: {len(train_dataloader)} "
        + f"batches with {train_dataloader.batch_size} images in each\n"
    )

    for epoch_number in range(1, image_classifier_config.training.epochs_count + 1):
        train_results = perform_training_iteration(
            neural_net, train_dataloader, optimizer, loss_function, accuracy_function
        )

        test_results = test_neural_net(
            neural_net,
            test_dataloader,
            accuracy_function,
            image_classifier_config.device,
        )

        click.echo(f"Epoch #{epoch_number} train results: {str(train_results)}")
        click.echo(f"\tTest results: {str(test_results)}\n")


def perform_training_iteration(
    neural_net: CustomConvolutionalNet,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    accuracy_function: Callable,
):
    neural_net.train()

    total_loss = 0
    total_accuracy = 0

    for batch in train_dataloader:
        images: torch.Tensor = batch[0].to(device=image_classifier_config.device)
        ideal_classes_output: torch.Tensor = batch[1].to(
            device=image_classifier_config.device
        )

        raw_output = neural_net(images)

        batch_loss = loss_function(raw_output, ideal_classes_output)
        total_loss += batch_loss.item()

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        predicted_probabilities: torch.Tensor = raw_output.softmax(0)
        predicted_classes = predicted_probabilities.argmax(dim=1)

        batch_accuracy = accuracy_function(predicted_classes, ideal_classes_output)
        total_accuracy += batch_accuracy.item()

    batches_count = len(train_dataloader)

    average_loss = total_loss / batches_count
    average_accuracy = (total_accuracy / batches_count) * 100

    train_results = NeuralNetMetrics(loss=average_loss, accuracy=average_accuracy)

    return train_results
