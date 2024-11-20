import numpy as np

from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.relu_gradient = None

    def forward(self, input_tensor):
        self.relu_gradient = np.maximum(input_tensor, 0.0)
        return self.relu_gradient

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)
