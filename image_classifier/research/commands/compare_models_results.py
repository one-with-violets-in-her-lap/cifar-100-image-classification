import json
import click
from matplotlib import pyplot as plt

from image_classifier.research.lib.metrics import NeuralNetMetrics
from image_classifier.config import image_classifier_config


@click.command("compare-models-results")
def compare_models_results():
    with open(
        image_classifier_config.model_results_file_path, "rt", encoding="utf-8"
    ) as models_results_json_stream:
        models_results_dicts: list[dict] = json.load(models_results_json_stream)
        models_results = [
            NeuralNetMetrics(**model_results) for model_results in models_results_dicts
        ]

        figure = plt.figure(figsize=(16, 9))

        # Loss bar chart
        figure.add_subplot(1, 2, 1)
        plt.title("Average loss")
        for results_record in models_results:
            plt.bar(results_record.neural_net_name, results_record.get_best_loss())

        # Accuracy bar chart
        figure.add_subplot(1, 2, 2)
        plt.title("Average accuracy")
        for results_record in models_results:
            plt.bar(
                results_record.neural_net_name, results_record.get_best_accuracy()
            )

        plt.show()
