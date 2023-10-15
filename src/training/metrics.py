from monai.utils import MetricReduction

from src.utils.constants import SEGMENTATION, CLASSIFICATION
import torch
from overrides import override
from monai.metrics import DiceMetric
from torch.nn import Softmax


class PipelineMetric:
    def step(self, prediction: torch.Tensor, label: torch.Tensor): ...
    def finish(self): ...


class Accuracy(PipelineMetric):
    def __init__(self):
        super().__init__()
        self.correct_count, self.total_items = 0, 0
        self.results_per_label = {}
        self.total_per_label = {}

    @override
    def step(self, prediction: torch.Tensor, label: torch.Tensor):
        """
        Given two tensors, counts the agreement using a one-hot encoding.
        :param prediction: batched predictions
        :param label: batched labels
        """
        assert prediction.shape == label.shape, f"Tensor a and b are different shapes. Got {prediction.shape} and {label.shape}"
        assert len(prediction.shape) == 2, f"Why is the prediction shape {prediction.shape}"
        # compute argmax
        label_argmax = torch.argmax(label, dim=1)
        prediction_argmax = torch.argmax(prediction, dim=1)
        results = prediction_argmax == label_argmax
        self.total_items += prediction.shape[0]
        # count correct per class
        for label, pred in zip(label_argmax, prediction_argmax):
            label = label.cpu().item()
            pred = pred.cpu().item()
            if label not in self.results_per_label:
                self.results_per_label[label] = 0
                self.total_per_label[label] = 0
            self.results_per_label[label] += (1 if label == pred else 0)
            self.total_per_label[label] += 1
        # case_distribution_fold
        self.correct_count += results.sum().item()

    @override
    def finish(self):
        """
        :return: Total correct, results per label, total per label
        """
        metric = self.correct_count/self.total_items, self.results_per_label, self.total_per_label
        self.correct_count, self.total_items = 0, 0
        self.results_per_label = {}
        self.total_per_label = {}
        return metric


class Dice(PipelineMetric):
    def __init__(self):
        self.total_items = 0.
        self.dice_metric = DiceMetric()
        self.softmax = Softmax(dim=1)

    @override
    def step(self, prediction: torch.Tensor, label: torch.Tensor):
        self.total_items += prediction.shape[0]
        prediction = self.softmax(prediction)
        prediction = torch.argmax(prediction, dim=1)
        self.dice_metric(y_pred=prediction, y=label)

    @override
    def finish(self):
        metric = self.dice_metric.aggregate()/self.total_items
        self.__init__()
        return metric
