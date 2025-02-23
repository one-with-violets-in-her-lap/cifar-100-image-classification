import torch
from torchvision import datasets
from torchvision import transforms

from image_classifier.utils.train_test_split import TrainTestValue


cifar_100_train_dataset = datasets.CIFAR100(
    "./datasets/cifar-100",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

cifar_100_test_dataset = datasets.CIFAR100(
    "./datasets/cifar-100",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)


def create_cifar_100_dataloaders(batch_size: int, num_workers: int):
    cifar_100_train_dataloader = torch.utils.data.DataLoader(
        cifar_100_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    cifar_100_test_dataloader = torch.utils.data.DataLoader(
        cifar_100_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return TrainTestValue(
        train=cifar_100_train_dataloader, test=cifar_100_test_dataloader
    )
