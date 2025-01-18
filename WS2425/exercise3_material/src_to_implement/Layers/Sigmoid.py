from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output = 1 / (1 + np.exp(-self.input_tensor))
        return output

    def backward(self, error_tensor):
        derivative = (
            1 / (1 + np.exp(-self.input_tensor)) * (1 - 1 / (1 + np.exp(-self.input_tensor)))
        )
        prev_error_tensor = error_tensor * derivative
        return prev_error_tensor
