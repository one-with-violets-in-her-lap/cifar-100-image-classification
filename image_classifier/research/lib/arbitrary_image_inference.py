import typing
import torch
from torchvision import transforms
from PIL import Image

from image_classifier.models.res_net import ResNet18
from image_classifier.data.cifar_100 import cifar_100_test_dataset
from image_classifier.models.test_time_augmentation import enable_test_time_augmentation
from image_classifier.train.lib.training_checkpoint import TrainingCheckpoint


arbitrary_image_transforms = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def classify_image(image: Image.Image, device: str, training_checkpoint_path: str):
    image_tensor = typing.cast(torch.Tensor, arbitrary_image_transforms(image))
    image_tensor = image_tensor.to(device=device)

    neural_net = ResNet18(len(cifar_100_test_dataset.classes))

    training_checkpoint: TrainingCheckpoint = torch.load(training_checkpoint_path)
    neural_net.load_state_dict(training_checkpoint["neural_net_state_dict"])

    neural_net = enable_test_time_augmentation(neural_net)
    neural_net.to(device=device)

    neural_net.eval()
    with torch.inference_mode():
        raw_output: torch.Tensor = neural_net(image_tensor.unsqueeze(dim=0))

        predicted_probabilities: torch.Tensor = raw_output.softmax(dim=1)
        predicted_class = predicted_probabilities.argmax(dim=1)

        return predicted_class.squeeze()
