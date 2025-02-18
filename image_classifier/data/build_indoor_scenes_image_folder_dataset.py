"""
Builds a standard image folder dataset from Indoor Scenes CVPR 2019 dataset with .txt
files containing paths
"""

import os
from pathlib import Path
from shutil import copyfile

import click

from image_classifier.utils.click_cli import start_click_cli_with_pretty_errors


SOURCE_DATASET_ROOT_PATH = "./datasets/indoor-scenes-cvpr-2019"
SOURCE_DATASET_IMAGES_PATH = "./datasets/indoor-scenes-cvpr-2019/indoorCVPR_09/Images"

TRANSFORMED_DATASET_ROOT_PATH = "./datasets/indoor-scenes-image-folder"


@click.command()
def build_dataset():
    if not os.path.isdir(SOURCE_DATASET_ROOT_PATH):
        raise FileNotFoundError(
            f"Indoor scenes dataset folder cannot be found at `{SOURCE_DATASET_ROOT_PATH}`"
        )

    with open(
        os.path.join(SOURCE_DATASET_ROOT_PATH, "TestImages.txt"), "rt", encoding="utf-8"
    ) as test_images_list_stream, open(
        os.path.join(SOURCE_DATASET_ROOT_PATH, "TrainImages.txt"),
        "rt",
        encoding="utf-8",
    ) as train_images_list_stream:
        test_images_paths = test_images_list_stream.read().split("\n")
        train_images_paths = train_images_list_stream.read().split("\n")

    TRAIN_IMAGES_FOLDER_PATH = os.path.join(TRANSFORMED_DATASET_ROOT_PATH, "train")
    TEST_IMAGES_FOLDER_PATH = os.path.join(TRANSFORMED_DATASET_ROOT_PATH, "test")

    os.makedirs(TRAIN_IMAGES_FOLDER_PATH, exist_ok=True)
    os.makedirs(TEST_IMAGES_FOLDER_PATH, exist_ok=True)

    def copy_dataset_images(images_paths: list[str], destination_folder_path: str):
        for image_path in images_paths:
            image_full_path = os.path.join(SOURCE_DATASET_IMAGES_PATH, image_path)

            new_image_path = os.path.join(destination_folder_path, image_path)
            new_image_folder_path = str(Path(new_image_path).parent)

            click.echo(
                click.style(image_full_path, dim=True)
                + click.style(" copied to -> ", bold=True)
                + click.style(new_image_folder_path, dim=True)
            )

            os.makedirs(new_image_folder_path, exist_ok=True)
            copyfile(image_full_path, new_image_path)

    copy_dataset_images(train_images_paths, TRAIN_IMAGES_FOLDER_PATH)
    copy_dataset_images(test_images_paths, TEST_IMAGES_FOLDER_PATH)

    click.echo(
        "\n"
        + f"Copied {len(train_images_paths)} train samples and "
        + f"{len(test_images_paths)} test samples to {TRANSFORMED_DATASET_ROOT_PATH}"
    )


if __name__ == "__main__":
    start_click_cli_with_pretty_errors(build_dataset)
