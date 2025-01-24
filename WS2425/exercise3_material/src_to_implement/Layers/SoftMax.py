import numpy as np # type: ignore

from Layers.Base import BaseLayer, PhaseSeperatableLayer

class SoftMax(PhaseSeperatableLayer):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_tensor):
        # if x_k > 0 --> e^(x_k) might become very large
        # So, shifting to increase numerical stability: x_hat = x - max(x)
        scaled_input = input_tensor - np.max(input_tensor, axis = 1, keepdims = True) 

        exps = np.exp(scaled_input) 
        self.y_hat = exps / np.sum(exps, axis = 1, keepdims = True)
        return self.y_hat
    
    def backward(self, error_tensor):
        # E_(n-1) = y_hat * (E_n - E_n * y_hat)
        return self.y_hat * (
            error_tensor - (error_tensor * self.y_hat).sum(axis = 1).reshape(-1, 1))
