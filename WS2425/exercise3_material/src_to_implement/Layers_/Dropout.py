from Layers.Base import BaseLayer
import numpy as np


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask_cache = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        # Inverted dropout
        self.mask_cache = (1 / self.probability) * np.random.choice(
            [0, 1], size=input_tensor.shape, p=[1 - self.probability, self.probability]
        )
        return input_tensor * self.mask_cache

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        return error_tensor * self.mask_cache