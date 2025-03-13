import click

from image_classifier.config import image_classifier_config
from image_classifier.research.commands.compare_models_results import (
    handle_compare_models_results_command,
)
from image_classifier.research.commands.plot_loss_and_accuracy_curves import (
    handle_plot_and_accuracy_command,
)
from image_classifier.utils.seeding import set_seed_for_randomness
from image_classifier.data.commands.build_indoor_scenes_image_folder_dataset import (
    handle_build_dataset_command,
)
from image_classifier.research.commands.view_cifar_100_dataset import (
    handle_view_cifar_100_dataset_command,
)
from image_classifier.train.commands.train import (
    handle_train_command,
)
from image_classifier.test.commands.test_model_from_checkpoint import (
    handle_test_model_command,
)


@click.group()
def cli():
    if image_classifier_config.seed is not None:
        set_seed_for_randomness(image_classifier_config.seed)


cli.add_command(handle_build_dataset_command)
cli.add_command(handle_view_cifar_100_dataset_command)
cli.add_command(handle_compare_models_results_command)
cli.add_command(handle_train_command)
cli.add_command(handle_test_model_command)
cli.add_command(handle_plot_and_accuracy_command)
