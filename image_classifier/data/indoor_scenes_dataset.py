import os

import click


_DATASET_ROOT_FOLDER_PATH = "./datasets/indoor-scenes-cvpr-2019"
_DATASET_TRAIN_IMAGES_LIST_PATH = os.path.join(
    _DATASET_ROOT_FOLDER_PATH, "TrainImages.txt"
)
_DATASET_TEST_IMAGES_LIST_PATH = os.path.join(
    _DATASET_ROOT_FOLDER_PATH, "TestImages.txt"
)

with open(
    _DATASET_TRAIN_IMAGES_LIST_PATH, "rt", encoding="utf-8"
) as train_images_stream, open(
    _DATASET_TEST_IMAGES_LIST_PATH, "rt", encoding="utf-8"
) as test_images_stream:
    _train_images_paths = train_images_stream.read().split("\n")
    _test_images_paths = test_images_stream.read().split("\n")

click.echo(
    f"Loading indoor scenes image dataset with {len(_train_images_paths)} train "
    + f"samples and {len(_test_images_paths)} test samples"
)
