import click
from matplotlib import pyplot as plt
from torchvision import io, transforms
from PIL import Image

from image_classifier.common_errors import CheckpointFilePathNotSpecifiedError
from image_classifier.config import image_classifier_config
from image_classifier.data.cifar_100 import cifar_100_test_dataset
from image_classifier.research.lib.arbitrary_image_inference import classify_image


@click.command("classify")
@click.option("-i", "--image-path", required=True)
def handle_arbitrary_image_classification_command(image_path: str):
    image = Image.open(image_path)

    click.echo(
        f"Loaded the image ({image.width}x{image.height}), "
        + "running the inference..."
    )

    if image_classifier_config.training_checkpoint_path is None:
        raise CheckpointFilePathNotSpecifiedError()

    image_predicted_class_index = classify_image(
        image,
        image_classifier_config.device,
        image_classifier_config.training_checkpoint_path,
    )

    predicted_class = cifar_100_test_dataset.classes[image_predicted_class_index]

    click.echo(
        f"Predicted image class: {image_predicted_class_index} - {predicted_class}"
    )

    plt.figure(figsize=(14, 6))
    plt.imshow(image)
    plt.title(f"{image_predicted_class_index} - {predicted_class}")
    plt.show()
