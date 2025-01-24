from Layers.Base import BaseLayer
import numpy as np


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.tanh(self.input_tensor)

    def backward(self, error_tensor):
        return error_tensor * (1 - np.tanh(self.input_tensor) ** 2)
