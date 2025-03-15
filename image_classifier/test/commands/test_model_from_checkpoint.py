import click
import torch
from torchmetrics import Accuracy
from torch import nn

from image_classifier.common_errors import CheckpointFilePathNotSpecifiedError
from image_classifier.config import image_classifier_config
from image_classifier.data.cifar_100 import (
    create_cifar_100_dataloaders,
    cifar_100_test_dataset,
)
from image_classifier.models.res_net import ResNet18
from image_classifier.test.lib.test import test_neural_net
from image_classifier.train.lib.training_checkpoint import TrainingCheckpoint


@click.command("test")
def handle_test_model_command():
    test_dataloader = create_cifar_100_dataloaders(
        image_classifier_config.batch_size, image_classifier_config.num_workers
    ).test

    neural_net = ResNet18(classes_count=len(cifar_100_test_dataset.classes))
    neural_net.to(device=image_classifier_config.device)

    if image_classifier_config.training_checkpoint_path is None:
        raise CheckpointFilePathNotSpecifiedError()

    training_checkpoint: TrainingCheckpoint = torch.load(
        image_classifier_config.training_checkpoint_path
    )

    neural_net.load_state_dict(training_checkpoint["neural_net_state_dict"])

    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device=image_classifier_config.device)

    accuracy_function = Accuracy(
        "multiclass", num_classes=len(cifar_100_test_dataset.classes)
    )
    accuracy_function.to(device=image_classifier_config.device)

    click.echo("Evaluating model from a checkpoint on CIFAR-100 test dataset")

    results = test_neural_net(
        neural_net,
        test_dataloader,
        accuracy_function,
        loss_function,
        image_classifier_config.device,
    )

    click.echo(click.style("\nResults", bold=True) + "\n" + str(results))
