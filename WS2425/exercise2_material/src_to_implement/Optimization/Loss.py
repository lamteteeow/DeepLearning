import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        # Ignore the prediction tensor with label = 0:
        prediction_tensor_1 = prediction_tensor * label_tensor
        # New array with prediction with label y = 1:
        prediction_tensor_1 = np.sum(prediction_tensor_1, axis=1)
        # Calculate loss for yk = 1:
        eps = np.finfo(np.float64).eps
        loss = np.sum(-np.log(prediction_tensor_1 + eps))
        # Save prediction tensor to calculate backward:
        self.prediction_tensor = prediction_tensor
        return loss

    def backward(self, label_tensor):
        # Calculate error tensor:
        eps = np.finfo(np.float64).eps
        prev_error_tensor = -label_tensor / (self.prediction_tensor + eps)
        return prev_error_tensor
