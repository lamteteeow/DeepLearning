import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.random((input_size + 1, output_size))
        self._optimizer = None
        self.gradient_weights = None
        self.input_tensor = None

    def forward(self, input_tensor):
        b_input_tensor = np.append(
            input_tensor, np.ones([input_tensor.shape[0], 1]), axis=1
        )
        self.input_tensor = b_input_tensor
        return np.dot(b_input_tensor, self.weights)

    def backward(self, error_tensor):
        back_weights = self.weights[:-1]
        curr_error_tensor = np.dot(error_tensor, b=back_weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.get_optimizer() is not None:
            self.weights = self.get_optimizer().calculate_update(
                self.weights, self.gradient_weights
            )
        return curr_error_tensor

    def get_optimizer(self):
        return self._optimizer

    # def set_optimizer(self, learning_rate):
    #     FullyConnected._optimizer = Sgd(learning_rate)
    #     return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)
