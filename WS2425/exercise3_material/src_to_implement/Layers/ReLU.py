import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)
    
    def backward(self, error_tensor):
        pre_error_tensor = error_tensor * (self.input_tensor > 0)
        return pre_error_tensor