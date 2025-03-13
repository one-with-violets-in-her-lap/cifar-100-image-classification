import math
import random

import click
from matplotlib import pyplot as plt
import torch

from image_classifier.data.indoor_scenes_dataset import indoor_scenes_train_dataset


_IMAGES_TO_SHOW_COUNT = 4


@click.command("view-indoor-scenes-dataset")
def handle_view_indoor_scenes_dataset_command():
    figure = plt.figure(figsize=(14, 6))
    plt.title("Indoor Scenes CVPR 2019")
    plt.axis(False)

    for number in range(1, _IMAGES_TO_SHOW_COUNT + 1):
        grid_axes_size = math.ceil(_IMAGES_TO_SHOW_COUNT / 2)
        figure.add_subplot(grid_axes_size, grid_axes_size, number)

        random_sample_index = random.randint(0, len(indoor_scenes_train_dataset) - 1)
        random_sample = indoor_scenes_train_dataset[random_sample_index]

        random_image: torch.Tensor = random_sample[0]
        random_image_class_index: int = random_sample[1]

        random_image_class = indoor_scenes_train_dataset.classes[
            random_image_class_index
        ]

        plt.title(f"{number}. {random_image_class}")
        plt.axis(False)
        plt.imshow(random_image.permute(1, 2, 0))

    plt.show()
