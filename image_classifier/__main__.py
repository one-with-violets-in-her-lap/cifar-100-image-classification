import click

from image_classifier.config import image_classifier_config
from image_classifier.data.commands.build_indoor_scenes_image_folder_dataset import (
    build_dataset,
)
from image_classifier.research.commands.view_indoor_scenes_dataset import (
    view_indoor_scenes_dataset,
)
from image_classifier.utils.seeding import set_seed_for_randomness


@click.group()
def cli():
    if image_classifier_config.seed is not None:
        set_seed_for_randomness(image_classifier_config.seed)


cli.add_command(build_dataset)
cli.add_command(view_indoor_scenes_dataset)

if __name__ == "__main__":
    cli()
