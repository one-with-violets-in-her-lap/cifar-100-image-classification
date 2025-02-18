import torch
from torchvision.transforms.transforms import random


def set_seed_for_randomness(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
