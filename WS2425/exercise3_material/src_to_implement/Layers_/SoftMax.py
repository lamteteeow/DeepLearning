from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        x_tilde = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        input_tensor_exp = np.exp(x_tilde)  # array of exp of each element
        input_tensor_sum = np.sum(input_tensor_exp, axis=1, keepdims=True)
        self.next_input_tensor = np.divide(input_tensor_exp, input_tensor_sum)
        return self.next_input_tensor

    def backward(self, error_tensor):
        # Calculate error tensor for previous layer:
        return self.next_input_tensor * (
            error_tensor - np.sum(error_tensor * self.next_input_tensor, axis=1, keepdims=True)
        )