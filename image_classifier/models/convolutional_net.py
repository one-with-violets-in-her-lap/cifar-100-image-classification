import click
import torch
from torch import nn


class ConvolutionalNet(nn.Module):
    def __init__(self, classes_count: int):
        super().__init__()

        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

        self.relu_layer = nn.ReLU()

        self.conv_layer_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_layer = nn.Flatten()

        self.fully_connected_layer_1 = nn.Linear(
            in_features=16 * 13 * 13, out_features=120
        )
        self.fully_connected_layer_2 = nn.Linear(in_features=120, out_features=64)
        self.fully_connected_layer_3 = nn.Linear(
            in_features=64, out_features=classes_count
        )

        self.layers = nn.Sequential(
            self.conv_layer_1,
            self.max_pool_layer,
            self.conv_layer_2,
            self.flatten_layer,
            self.fully_connected_layer_1,
            self.fully_connected_layer_2,
            self.fully_connected_layer_3,
        )

    def forward(self, input_data: torch.Tensor):
        output: torch.Tensor = self.conv_layer_1(input_data)
        click.echo(f"Conv layer 1 output: {output.shape}")
        output = self.relu_layer(output)
        output = self.max_pool_layer(output)

        output = self.conv_layer_2(output)
        output = self.relu_layer(output)
        output = self.max_pool_layer(output)
        click.echo(f"Conv layer 2 output with pooling and activation: {output.shape}")

        output = self.flatten_layer(output)
        click.echo(f"Flatten layer output: {output.shape}")

        output = self.fully_connected_layer_1(output)
        click.echo(f"FC layer 1 output: {output.shape}")
        output = self.relu_layer(output)

        output = self.fully_connected_layer_2(output)
        click.echo(f"FC layer 2 output: {output.shape}")
        output = self.relu_layer(output)

        output = self.fully_connected_layer_3(output)
        click.echo(f"FC layer 3 output: {output.shape}")

        return output
