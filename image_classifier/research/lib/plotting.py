from matplotlib import pyplot as plt

from image_classifier.research.lib.metrics import NeuralNetMetrics


def plot_loss_and_accuracy(*metrics_items: NeuralNetMetrics):
    figure = plt.figure(figsize=(16, 9))

    current_subplot_index = 0

    for metrics in metrics_items:
        # Loss curve
        current_subplot_index += 1
        figure.add_subplot(len(metrics_items), 2, current_subplot_index)
        plt.title(f"{metrics.neural_net_name} | Loss")

        plt.plot(metrics.loss_records_for_each_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Accuracy curve
        current_subplot_index += 1
        figure.add_subplot(len(metrics_items), 2, current_subplot_index)
        plt.title(f"{metrics.neural_net_name} | Accuracy")

        plt.plot(metrics.accuracy_records_for_each_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

    plt.show()
