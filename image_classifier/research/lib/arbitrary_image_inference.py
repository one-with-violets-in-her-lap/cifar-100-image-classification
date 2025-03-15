import torch
from torchvision import transforms

from image_classifier.models.res_net import ResNet18
from image_classifier.data.cifar_100 import cifar_100_test_dataset
from image_classifier.train.lib.training_checkpoint import TrainingCheckpoint


arbitrary_image_transforms = transforms.Compose([transforms.Resize((32, 32))])


def classify_image(image: torch.Tensor, device: str, training_checkpoint_path: str):
    image = image.to(device=device, dtype=torch.float)

    image = arbitrary_image_transforms(image)

    neural_net = ResNet18(len(cifar_100_test_dataset.classes))
    neural_net.to(device=device)

    training_checkpoint: TrainingCheckpoint = torch.load(training_checkpoint_path)
    neural_net.load_state_dict(training_checkpoint["neural_net_state_dict"])

    neural_net.eval()
    with torch.inference_mode():
        raw_output: torch.Tensor = neural_net(image.unsqueeze(dim=0))

        predicted_probabilities: torch.Tensor = raw_output.softmax(dim=1)
        predicted_class = predicted_probabilities.argmax(dim=1)

        print(raw_output, predicted_probabilities)

        return predicted_class.squeeze()
