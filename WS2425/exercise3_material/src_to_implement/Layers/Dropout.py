from Layers.Initializers import BaseInitializer
import numpy as np #type: ignore
from Layers.Base import PhaseSeperatableLayer

class Dropout(PhaseSeperatableLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor

        self.mask = (np.random.rand(*input_tensor.shape) < self.probability).astype(float)
        self.mask = self.mask / self.probability
        return input_tensor * self.mask

    def backward(self, error_tensor):
        return error_tensor * self.mask

