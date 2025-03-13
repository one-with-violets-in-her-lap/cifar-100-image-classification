import math
import random

import click
from matplotlib import pyplot as plt
import torch

from image_classifier.data.cifar_100 import cifar_100_train_dataset


_IMAGES_TO_SHOW_COUNT = 4


@click.command("view-cifar-100-dataset")
def handle_view_cifar_100_dataset_command():
    figure = plt.figure(figsize=(14, 6))
    plt.title("CIFAR-100")
    plt.axis(False)

    for number in range(1, _IMAGES_TO_SHOW_COUNT + 1):
        grid_axes_size = math.ceil(_IMAGES_TO_SHOW_COUNT / 2)
        figure.add_subplot(grid_axes_size, grid_axes_size, number)

        random_sample_index = random.randint(0, len(cifar_100_train_dataset) - 1)
        random_sample = cifar_100_train_dataset[random_sample_index]

        random_image: torch.Tensor = random_sample[0]
        random_image_class_index: int = random_sample[1]

        random_image_class = cifar_100_train_dataset.classes[random_image_class_index]

        plt.title(f"{number}. {random_image_class}")
        plt.axis(False)
        plt.imshow(random_image.permute(1, 2, 0))

    plt.show()
