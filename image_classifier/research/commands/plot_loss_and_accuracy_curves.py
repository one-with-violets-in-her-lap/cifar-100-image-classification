import json

import click

from image_classifier.config import image_classifier_config
from image_classifier.research.lib.metrics import (
    NeuralNetTrainTestMetrics,
)
from image_classifier.research.lib.plotting import plot_loss_and_accuracy
from image_classifier.utils.dataclass_from_dict import (
    create_dataclass_instance_from_dict,
)


@click.command("plot-loss-and-accuracy")
def handle_plot_and_accuracy_command():
    with open(
        image_classifier_config.training.models_results_file_path, "rt", encoding="utf-8"
    ) as models_results_json_stream:
        models_results_dicts: list[dict] = json.load(models_results_json_stream)
        models_results = [
            create_dataclass_instance_from_dict(
                NeuralNetTrainTestMetrics, model_results
            )
            for model_results in models_results_dicts
        ]

        plot_loss_and_accuracy(*models_results)
