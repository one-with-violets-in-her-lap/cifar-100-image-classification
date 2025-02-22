import click
import torch
from torch import nn

from image_classifier.models.named_neural_net import NamedNeuralNet


class CustomConvolutionalNet(NamedNeuralNet):
    def __init__(self, classes_count: int):
        super().__init__('Custom convolutional network')

        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convolutional_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=classes_count),
        )

    def forward(self, input_data: torch.Tensor):
        output = self.convolutional_block_1(input_data)

        output = self.convolutional_block_2(output)

        output = self.convolutional_block_3(output)

        # click.echo(f'Third conv block output shape: {output.shape}')

        output = self.fully_connected_layers(output)

        return output
