import os
import click
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from image_classifier.utils.train_test_split import DatasetSplit, TrainTestValue


INDOOR_SCENES_IMAGE_FOLDER_DATASET = "./datasets/indoor-scenes-image-folder"

indoor_scenes_train_dataset = datasets.ImageFolder(
    root=os.path.join(INDOOR_SCENES_IMAGE_FOLDER_DATASET, DatasetSplit.TRAIN.value),
    transform=transforms.Compose(
        [transforms.Resize(size=(72, 72)), transforms.ToTensor()]
    ),
)

indoor_scenes_test_dataset = datasets.ImageFolder(
    root=os.path.join(INDOOR_SCENES_IMAGE_FOLDER_DATASET, DatasetSplit.TEST.value),
    transform=transforms.Compose(
        [
            transforms.Resize(size=(72, 72)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ]
    ),
)


def create_indoor_scenes_dataloaders(batch_size=32):
    click.echo("Initializing indoor scenes dataset dataloaders")

    indoor_scenes_train_dataloader = DataLoader(
        indoor_scenes_train_dataset, batch_size=batch_size, shuffle=True
    )
    indoor_scenes_test_dataloader = DataLoader(
        indoor_scenes_test_dataset, batch_size=batch_size, shuffle=False
    )

    return TrainTestValue(
        train=indoor_scenes_train_dataloader, test=indoor_scenes_test_dataloader
    )
