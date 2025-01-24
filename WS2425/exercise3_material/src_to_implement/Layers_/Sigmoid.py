from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    # Original implementation
    # def forward(self, input_tensor):
    #     self.input_tensor = input_tensor
    #     return 1 / (1 + np.exp(-self.input_tensor))
    #
    # def backward(self, error_tensor):
    #     derivative = (
    #         1 / (1 + np.exp(-self.input_tensor)) * (1 - 1 / (1 + np.exp(-self.input_tensor)))
    #     )
    #     return error_tensor * derivative

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return self.sigmoid(input_tensor)

    def backward(self, error_tensor):
        derivative = self.sigmoid(self.input_tensor) * (1 - self.sigmoid(self.input_tensor))
        return error_tensor * derivative
