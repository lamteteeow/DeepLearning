from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.relu_gradient = None

    def forward(self, input_tensor):
        output = np.maximum(input_tensor, 0.0)
        self.relu_gradient = output
        return output

    def backward(self, error_tensor):
        output = error_tensor * (self.relu_gradient > 0)
        return output
