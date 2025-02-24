import atexit
import json
import os
from typing import Callable
import click
import torch
from torch import nn
from torchmetrics import Accuracy

from image_classifier.data.cifar_100 import (
    create_cifar_100_dataloaders,
    cifar_100_train_dataset,
)
from image_classifier.config import image_classifier_config
from image_classifier.models.named_neural_net import NamedNeuralNet
from image_classifier.models.res_net import ResNet18
from image_classifier.research.metrics import NeuralNetMetrics
from image_classifier.train.test import test_neural_net


@click.command()
def train():
    dataloaders = create_cifar_100_dataloaders(
        image_classifier_config.training.batch_size,
        image_classifier_config.training.num_workers,
    )
    train_dataloader = dataloaders.train
    test_dataloader = dataloaders.test

    neural_net = ResNet18(classes_count=len(cifar_100_train_dataset.classes))
    neural_net.to(device=image_classifier_config.device)

    atexit.register(
        lambda: save_model(
            neural_net, image_classifier_config.saved_models_directory_path
        )
    )

    optimizer = torch.optim.SGD(
        neural_net.parameters(), lr=image_classifier_config.training.learning_rate
    )

    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
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

    highest_test_accuracy = 0
    lowest_test_loss = 0

    for epoch_number in range(1, image_classifier_config.training.epochs_count + 1):
        train_results = perform_training_iteration(
            neural_net, train_dataloader, optimizer, loss_function, accuracy_function
        )

        learning_rate_scheduler.step()

        test_results = test_neural_net(
            neural_net,
            test_dataloader,
            accuracy_function,
            image_classifier_config.device,
        )

        if highest_test_accuracy < test_results.accuracy:
            highest_test_accuracy = test_results.accuracy

        if lowest_test_loss > test_results.loss or lowest_test_loss == 0:
            lowest_test_loss = test_results.loss

        click.echo(f"Epoch #{epoch_number} train results: {str(train_results)}")
        click.echo(f"\tTest results: {str(test_results)}")
        click.echo(
            f"\tCurrent learning rate: {learning_rate_scheduler.get_last_lr()}\n"
        )

    click.echo("Finished training. Saving model results")

    save_model_results(
        NeuralNetMetrics(
            neural_net_name=neural_net.name,
            loss=lowest_test_loss,
            accuracy=highest_test_accuracy,
        )
    )


def perform_training_iteration(
    neural_net: NamedNeuralNet,
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

    train_results = NeuralNetMetrics(
        loss=average_loss, accuracy=average_accuracy, neural_net_name=neural_net.name
    )

    return train_results


def save_model_results(results: NeuralNetMetrics):
    models_results_dicts: list[dict] = []

    if os.path.exists(image_classifier_config.model_results_file_path):
        with open(
            image_classifier_config.model_results_file_path,
            "rt",
            encoding="utf-8",
        ) as models_results_json_stream:
            models_results_dicts = json.load(models_results_json_stream)

            models_results_dicts = [
                model_results
                for model_results in models_results_dicts
                if model_results["neural_net_name"] != results.neural_net_name
            ]

    models_results_dicts.append(vars(results))

    with open(
        image_classifier_config.model_results_file_path, "wt", encoding="utf-8"
    ) as models_results_json_write_stream:
        models_results_json_write_stream.write(json.dumps(models_results_dicts))

    click.echo(
        f"Model results saved to {image_classifier_config.model_results_file_path}"
    )


def save_model(model: NamedNeuralNet, saved_models_directory_path: str):
    if not os.path.exists(saved_models_directory_path):
        os.makedirs(saved_models_directory_path, exist_ok=True)

    path_to_save_model_to = os.path.join(
        saved_models_directory_path, f"{model.name}.pth"
    )

    torch.save(model.state_dict(), path_to_save_model_to)
    click.echo(f"Model saved to {path_to_save_model_to}")
