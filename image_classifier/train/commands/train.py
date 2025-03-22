import atexit
from dataclasses import asdict
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
from image_classifier.research.lib.metrics import (
    NeuralNetMetrics,
    NeuralNetTrainTestMetrics,
)
from image_classifier.train.lib.training_checkpoint import TrainingCheckpoint
from image_classifier.test.lib.test import test_neural_net
from image_classifier.utils.train_test_split import TrainTestValue


@click.command("train")
def handle_train_command():
    dataloaders = create_cifar_100_dataloaders(
        image_classifier_config.batch_size,
        image_classifier_config.num_workers,
    )
    train_dataloader = dataloaders.train
    test_dataloader = dataloaders.test

    neural_net = ResNet18(classes_count=len(cifar_100_train_dataset.classes))
    neural_net.to(device=image_classifier_config.device)

    optimizer = torch.optim.SGD(
        neural_net.parameters(),
        lr=image_classifier_config.training.learning_rate,
        momentum=image_classifier_config.training.momentum,
        weight_decay=image_classifier_config.training.weight_decay,
    )

    # Loads training checkpoint if it has been saved previously
    training_checkpoint = (
        load_training_checkpoint(image_classifier_config.training_checkpoint_path)
        if image_classifier_config.training_checkpoint_path is not None
        else None
    )

    if training_checkpoint is not None:
        optimizer.load_state_dict(training_checkpoint["optimizer_state_dict"])
        neural_net.load_state_dict(training_checkpoint["neural_net_state_dict"])

        click.echo(
            "Loaded training checkpoint from "
            + f"{image_classifier_config.training_checkpoint_path}\n"
        )

    # Registers the exit handler that saves a training checkpoint
    if image_classifier_config.training.save_training_checkpoint_on_exit:
        atexit.register(
            lambda: save_training_checkpoint(
                {
                    "neural_net_name": neural_net.name,
                    "neural_net_state_dict": neural_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                image_classifier_config.saved_models_directory_path,
            )
        )

    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=image_classifier_config.training.epochs_count - 20
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

    metrics = NeuralNetTrainTestMetrics(
        neural_net_name=neural_net.name,
        loss_records_for_each_epoch=TrainTestValue(test=[], train=[]),
        accuracy_records_for_each_epoch=TrainTestValue(test=[], train=[]),
    )

    if image_classifier_config.training.save_model_results_on_exit:
        atexit.register(lambda: save_model_results(metrics))

    for epoch_number in range(1, image_classifier_config.training.epochs_count + 1):
        train_results = perform_training_iteration(
            neural_net, train_dataloader, optimizer, loss_function, accuracy_function
        )

        learning_rate_scheduler.step()

        test_results = test_neural_net(
            neural_net,
            test_dataloader,
            accuracy_function,
            loss_function,
            image_classifier_config.device,
        )

        metrics.accuracy_records_for_each_epoch.train.append(
            train_results.get_best_accuracy()
        )
        metrics.loss_records_for_each_epoch.train.append(train_results.get_best_loss())

        metrics.accuracy_records_for_each_epoch.test.append(
            test_results.get_best_accuracy()
        )
        metrics.loss_records_for_each_epoch.test.append(test_results.get_best_loss())

        click.echo(f"Epoch #{epoch_number} train results: {str(train_results)}")
        click.echo(f"\tTest results: {str(test_results)}")
        click.echo(
            "\tCurrent learning rate: "
            + f"{optimizer.state_dict()['param_groups'][0]['lr']}\n"
        )

    click.echo("Finished training\n")


def perform_training_iteration(
    neural_net: NamedNeuralNet,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    accuracy_function: Callable,
):
    neural_net.train()

    train_results = NeuralNetMetrics(
        neural_net_name=neural_net.name,
        accuracy_records_for_each_epoch=[],
        loss_records_for_each_epoch=[],
    )

    for batch in train_dataloader:
        images: torch.Tensor = batch[0].to(device=image_classifier_config.device)
        ideal_classes_output: torch.Tensor = batch[1].to(
            device=image_classifier_config.device
        )

        raw_output = neural_net(images)

        batch_loss: torch.Tensor = loss_function(raw_output, ideal_classes_output)
        train_results.loss_records_for_each_epoch.append(batch_loss.item())

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        predicted_probabilities: torch.Tensor = raw_output.softmax(dim=1)
        predicted_classes = predicted_probabilities.argmax(dim=1)

        batch_accuracy: torch.Tensor = accuracy_function(
            predicted_classes, ideal_classes_output
        )
        train_results.accuracy_records_for_each_epoch.append(
            batch_accuracy.item() * 100
        )

    return train_results


def save_model_results(results: NeuralNetTrainTestMetrics):
    models_results_dicts: list[dict] = []

    if os.path.exists(image_classifier_config.training.models_results_file_path):
        with open(
            image_classifier_config.training.models_results_file_path,
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
        image_classifier_config.training.models_results_file_path,
        "wt",
        encoding="utf-8",
    ) as models_results_json_write_stream:
        models_results_json_write_stream.write(json.dumps(models_results_dicts))

    click.echo(
        "Model results saved to "
        + image_classifier_config.training.models_results_file_path
    )


def save_training_checkpoint(
    checkpoint: TrainingCheckpoint, saved_models_directory_path: str
):
    if not os.path.exists(saved_models_directory_path):
        os.makedirs(saved_models_directory_path, exist_ok=True)

    path_to_save_to = os.path.join(
        saved_models_directory_path,
        f"{checkpoint['neural_net_name']}.pt",
    )

    torch.save(checkpoint, path_to_save_to)
    click.echo(f"Training checkpoint saved to {path_to_save_to}")


def load_training_checkpoint(checkpoint_path_to_load: str) -> TrainingCheckpoint:
    return torch.load(checkpoint_path_to_load)
