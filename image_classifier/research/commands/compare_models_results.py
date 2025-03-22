import click
from matplotlib import pyplot as plt

from image_classifier.research.lib.metrics import (
    NeuralNetMetrics,
    load_models_results,
)
from image_classifier.config import image_classifier_config


@click.command("compare-models-results")
def handle_compare_models_results_command():
    models_results = load_models_results(
        image_classifier_config.testing.models_results_file_path, NeuralNetMetrics
    )

    figure = plt.figure(figsize=(16, 9))

    # Loss bar chart
    figure.add_subplot(1, 2, 1)
    plt.title("Best test loss (lower=better)")
    for results_record in models_results:
        plt.bar(
            results_record.neural_net_name,
            results_record.get_best_loss(),
        )

    # Accuracy bar chart
    figure.add_subplot(1, 2, 2)
    plt.title("Best test accuracy (higher=better)")
    for results_record in models_results:
        plt.bar(
            results_record.neural_net_name,
            results_record.get_best_accuracy(),
        )

    plt.show()
