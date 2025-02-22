import click

from image_classifier.config import image_classifier_config
from image_classifier.utils.seeding import set_seed_for_randomness
from image_classifier.data.commands.build_indoor_scenes_image_folder_dataset import (
    build_dataset,
)
from image_classifier.research.commands.view_indoor_scenes_dataset import (
    view_indoor_scenes_dataset,
)
from image_classifier.research.commands.view_cifar_100_dataset import (
    view_cifar_100_dataset,
)
from image_classifier.train.commands.train import (
    train,
)


@click.group()
def cli():
    if image_classifier_config.seed is not None:
        set_seed_for_randomness(image_classifier_config.seed)


cli.add_command(build_dataset)
cli.add_command(view_cifar_100_dataset)
cli.add_command(train)

if __name__ == "__main__":
    cli()
