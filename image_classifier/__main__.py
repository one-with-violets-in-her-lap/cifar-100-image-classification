import click

from image_classifier.data.commands.build_indoor_scenes_image_folder_dataset import (
    build_dataset,
)
from image_classifier.research.commands.view_indoor_scenes_dataset import (
    view_indoor_scenes_dataset,
)
from image_classifier.utils.click_cli import start_click_cli_with_pretty_errors


@click.group()
def cli():
    pass


cli.add_command(build_dataset)
cli.add_command(view_indoor_scenes_dataset)

if __name__ == "__main__":
    start_click_cli_with_pretty_errors(cli)
