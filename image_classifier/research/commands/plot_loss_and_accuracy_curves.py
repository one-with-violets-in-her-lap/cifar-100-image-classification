import json

import click

from image_classifier.config import image_classifier_config
from image_classifier.research.lib.metrics import NeuralNetMetrics
from image_classifier.research.lib.plotting import plot_loss_and_accuracy


@click.command('plot-loss-and-accuracy')
def handle_plot_and_accuracy_command():
    with open(
        image_classifier_config.model_results_file_path, "rt", encoding="utf-8"
    ) as models_results_json_stream:
        models_results_dicts: list[dict] = json.load(models_results_json_stream)
        models_results = [
            NeuralNetMetrics(**model_results) for model_results in models_results_dicts
        ]

        plot_loss_and_accuracy(*models_results)
