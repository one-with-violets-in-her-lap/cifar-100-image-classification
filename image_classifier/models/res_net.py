from torch import nn
import torch

from image_classifier.models.named_neural_net import NamedNeuralNet


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.25),
        )

    def forward(self, input_data: torch.Tensor):
        return self.layers(input_data)


class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first=False):
        super().__init__()

        residual_intermediate_channels = out_channels // 4

        stride = 1

        # Downsampling via conv layer with stride=2 to fix shape mismatches when
        # summing input signal and output features. See the `forward` function
        self.is_projection = in_channels != out_channels
        self.projection_shortcut = None

        if self.is_projection:
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=2, padding=0
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(p=0.25),
            )
            stride = 2
            residual_intermediate_channels = in_channels // 2

        if first:
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(p=0.25),
            )
            stride = 1
            residual_intermediate_channels = in_channels

        self.layers = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=residual_intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            ConvBlock(
                in_channels=residual_intermediate_channels,
                out_channels=residual_intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(),
            ConvBlock(
                in_channels=residual_intermediate_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, input_data: torch.Tensor):
        identity = input_data

        features: torch.Tensor = self.layers(input_data)

        # Downsampling to fix shape mismatches
        if self.projection_shortcut is not None:
            identity = self.projection_shortcut(identity)

        # ResNet input signal mapping
        output = features + identity

        return output


class ResNet(NamedNeuralNet):
    def __init__(self, blocks_counts: list[int], classes_count: int, in_channels=3):
        super().__init__("ResNet")

        out_features = [256, 512, 1024, 2048]

        first_residual_block = BottleNeckResidualBlock(
            in_channels=64, out_channels=256, first=True
        )
        self.residual_blocks = nn.ModuleList([first_residual_block])

        for feature_index in range(len(out_features)):
            if feature_index > 0:
                previous_feature_value = out_features[feature_index - 1]

                self.residual_blocks.append(
                    BottleNeckResidualBlock(
                        in_channels=previous_feature_value,
                        out_channels=out_features[feature_index],
                    )
                )

            for _ in range(blocks_counts[feature_index] - 1):
                self.residual_blocks.append(
                    BottleNeckResidualBlock(
                        in_channels=out_features[feature_index],
                        out_channels=out_features[feature_index],
                    )
                )

        self.conv_layer_1 = ConvBlock(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.max_pool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.average_pool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.fully_connected_layer = nn.Linear(
            in_features=2048, out_features=classes_count
        )

        self.relu_layer = nn.ReLU()

    def forward(self, input_data: torch.Tensor):
        output = self.relu_layer(self.conv_layer_1(input_data))
        output = self.max_pool_layer(output)

        for residual_block in self.residual_blocks:
            output = residual_block(output)

        output = self.average_pool_layer(output)

        output = torch.flatten(output, start_dim=1)

        output = self.fully_connected_layer(output)

        return output


class ResNet18(ResNet):
    def __init__(self, classes_count: int):
        super().__init__([2, 2, 2, 2], classes_count, 3)
        self.name = "Resnet 18"


class ResNet50(ResNet):
    def __init__(self, classes_count: int):
        super().__init__([3, 4, 6, 3], classes_count, 3)
        self.name = "Resnet 50"


class ResNet101(ResNet):
    def __init__(self, classes_count: int):
        super().__init__([3, 4, 23, 3], classes_count, 3)
        self.name = "Resnet 101"
