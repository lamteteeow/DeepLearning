import numpy as np #type: ignore
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        self.activation = 1.0 / (1 + np.exp(-input_tensor))
        return self.activation
    
    def backward(self, error_tensor):
        return error_tensor * self.activation * (1 - self.activation)

