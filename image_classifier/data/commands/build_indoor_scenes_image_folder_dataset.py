"""
Builds a standard image folder dataset from Indoor Scenes CVPR 2019 dataset with .txt
files containing paths
"""

import os
from pathlib import Path
from shutil import copyfile

import click

from image_classifier.data.indoor_scenes_dataset import (
    INDOOR_SCENES_IMAGE_FOLDER_DATASET,
)


SOURCE_DATASET_ROOT_PATH = "./datasets/indoor-scenes-cvpr-2019"
SOURCE_DATASET_IMAGES_PATH = "./datasets/indoor-scenes-cvpr-2019/indoorCVPR_09/Images"


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


@click.command("build-dataset")
def handle_build_dataset_command():
    if not os.path.isdir(SOURCE_DATASET_ROOT_PATH):
        raise FileNotFoundError(
            "Indoor scenes dataset folder cannot be found "
            + f"at `{SOURCE_DATASET_ROOT_PATH}`"
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

    train_images_folder_path = os.path.join(INDOOR_SCENES_IMAGE_FOLDER_DATASET, "train")
    test_images_folder_path = os.path.join(INDOOR_SCENES_IMAGE_FOLDER_DATASET, "test")

    os.makedirs(train_images_folder_path, exist_ok=True)
    os.makedirs(test_images_folder_path, exist_ok=True)

    copy_dataset_images(train_images_paths, train_images_folder_path)
    copy_dataset_images(test_images_paths, test_images_folder_path)

    click.echo(
        "\n"
        + f"Copied {len(train_images_paths)} train samples and "
        + f"{len(test_images_paths)} test samples to {INDOOR_SCENES_IMAGE_FOLDER_DATASET}"
    )
