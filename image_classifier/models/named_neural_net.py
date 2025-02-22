from torch import nn


class NamedNeuralNet(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
