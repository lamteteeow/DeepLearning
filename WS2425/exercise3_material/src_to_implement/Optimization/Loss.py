from typing import Any
from abc import ABC, abstractmethod
import numpy as np # type: ignore

class BaseLoss(ABC):
    @abstractmethod
    def forward(self, predicted_label, true_label):
        pass

    @abstractmethod
    def backward(self, true_label):
        pass

class CrossEntropyLoss(BaseLoss):
    def __init__(self) -> None:
        pass

    def forward(self, predicted_label, true_label):
        y, self.y_hat  = true_label, predicted_label
        loss = -np.sum(y * np.log(self.y_hat + np.finfo(float).eps))
        return loss
    
    def backward(self, true_label):
        return -(true_label / (self.y_hat + np.finfo(float).eps))