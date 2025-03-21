import ttach as tta
from torch import nn


def enable_test_time_augmentation(neural_net: nn.Module):
    """
    Wraps passed `neural_net` in a [ttach](https://github.com/qubvel/ttach) module that
    enables TTA

    Returns:
        Neural net wrapped in TTA module
    """

    net_with_test_time_augment = tta.ClassificationTTAWrapper(
        neural_net,
        transforms=tta.Compose(
            [
                tta.Multiply([0.9, 1, 1.1]),
                tta.HorizontalFlip(),
            ]
        ),
    )

    return net_with_test_time_augment
