import torch


default_device = "cuda" if torch.cuda.is_available() else "cpu"
