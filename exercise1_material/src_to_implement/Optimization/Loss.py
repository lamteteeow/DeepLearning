import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        eps = np.finfo(float).eps
        loss = np.sum(-np.log(self.prediction_tensor + eps), where=(label_tensor == 1))
        return loss

    def backward(self, label_tensor):
        eps = np.finfo(float).eps
        return -label_tensor / (self.prediction_tensor + eps)
