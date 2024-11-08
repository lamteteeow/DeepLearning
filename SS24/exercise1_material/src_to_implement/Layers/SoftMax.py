import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        x_k = input_tensor
        x_tilde = x_k - np.max(x_k, axis=1, keepdims=True)
        input_tensor_exp = np.exp(x_tilde)  # array of exp of each element
        input_tensor_sum = np.sum(input_tensor_exp, axis=1, keepdims=True)
        next_input_tensor = np.divide(input_tensor_exp, input_tensor_sum)

        self.next_input_tensor = next_input_tensor
        return next_input_tensor

    def backward(self, error_tensor):
        # Calculate error tensor for previous layer:
        pre_error_tensor = self.next_input_tensor * (
            error_tensor
            - np.sum(error_tensor * self.next_input_tensor, axis=1, keepdims=True)
        )
        return pre_error_tensor
