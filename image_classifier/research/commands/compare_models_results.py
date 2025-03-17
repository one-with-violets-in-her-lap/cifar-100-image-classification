import json
import click
from matplotlib import pyplot as plt

from image_classifier.research.lib.metrics import (
    NeuralNetTrainTestMetrics,
)
from image_classifier.config import image_classifier_config
from image_classifier.utils.dataclass_from_dict import create_dataclass_instance_from_dict
from image_classifier.utils.train_test_split import DatasetSplit


@click.command("compare-models-results")
def handle_compare_models_results_command():
    with open(
        image_classifier_config.model_results_file_path, "rt", encoding="utf-8"
    ) as models_results_json_stream:
        models_results_dicts: list[dict] = json.load(models_results_json_stream)
        models_results = [
            create_dataclass_instance_from_dict(NeuralNetTrainTestMetrics, model_results)
            for model_results in models_results_dicts
        ]

        figure = plt.figure(figsize=(16, 9))

        # Loss bar chart
        figure.add_subplot(1, 2, 1)
        plt.title("Best test loss")
        for results_record in models_results:
            plt.bar(
                results_record.neural_net_name,
                results_record.get_best_loss(dataset_split=DatasetSplit.TEST),
            )

        # Accuracy bar chart
        figure.add_subplot(1, 2, 2)
        plt.title("Best test accuracy")
        for results_record in models_results:
            plt.bar(
                results_record.neural_net_name,
                results_record.get_best_accuracy(dataset_split=DatasetSplit.TEST),
            )

        plt.show()
