import numpy as np #type: ignore
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation
    
    def backward(self, error_tensor):
        return error_tensor * (1 - self.activation ** 2)