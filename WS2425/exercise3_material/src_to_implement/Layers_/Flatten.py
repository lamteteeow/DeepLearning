from Layers.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(self.input_shape[0], -1)

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)
