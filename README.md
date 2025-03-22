# CIFAR-100 image classification with ResNet

ResNet implementation trained on CIFAR-100 dataset. Achieved 73% test accuracy with
test-time augmentation applied

## Usage

### Try on Google Colab

The easiest way to just run a few predictions and try the model @
[Open Google Colab](https://colab.research.google.com/drive/1s9BRagS2_YA1jA3d6BZVapk51BSOXD0v?usp=drive_link)

### Running locally

Python 3.10 is recommended. Clone the repo and install dependencies:

```bash
# PIP
pip install .

# Poetry
poetry install
```

Download model weights from
[Github release](https://github.com/one-with-violets-in-her-lap/cifar-100-image-classification/releases/latest)
and put the file in `./bin` folder

Prepare some image and run the inference:

```sh
# PIP
python -m image_classifier.main classify --image-path ./image.jpg

# Poetry
poetry run image-classifier classify --image-path ./image.jpg
```

## Implementation details

| Basic info |  |
| --- | --- |
| **Neural net architecture** | ResNet 18 with bottleneck blocks |
| **Optimizer** | SGD with [CosineAnnelingLR scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) |
| **Optimizer params** | Initial learning rate: 0.02. Weight decay: 0.0005. Momentum: 0.9 |
| **Loss function** | Cross entropy loss |

| Data processing | |
  | --- | --- |
  | Dataset | CIFAR-100 |
| Batch size | 128 |
| Transforms | Upscaling to 64x64, applying `TrivialAugmentWide` and normalization |

Test-time augmentation is used for inference to increase test accuracy by 3 percent

![bar chart comparison between TTA (accuracy: 73%, loss: 0.975) and non-TTA (accuracy: 70%, loss: 1.141) inference](https://github.com/user-attachments/assets/0e3ea882-4cf7-48a3-a6fe-60f6dcd2a572)

## State of the model

The ~73% test accuracy is far from ideal. The model is currently overfitting, probably due
to shortage of data to train on

![line charts shows ~99 training accuracy curve and ~70 test accuracy curve](https://github.com/user-attachments/assets/209005dd-2f64-4fe1-b78d-fa359665a552)

The possible solution is to train on other additional larger dataset (e.g. ImageNet).
Switching to more performant neural net architecture is also a great idea (e.g. PyramidNet)

## References

- hf project with ResNet 18 hyperparams tuned for CIFAR-100 - https://huggingface.co/edadaltocg/resnet18_cifar100
- "Deep Residual Learning for Image Recognition" paper - https://arxiv.org/abs/1512.03385
- "How to Use Test-Time Augmentation to Make Better Predictions" - https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/
