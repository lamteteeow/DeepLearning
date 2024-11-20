import numpy as np
import Layers.Base as BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # self.input_shape = input_tensor.shape[0]
        # self.output_shape = (1, input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3])
        # return input_tensor.reshape(self.output_shape)

        self.batch_size = input_tensor.shape[0]
        self.input_shape = input_tensor.shape[1:10]

        return input_tensor.reshape(self.batch_size, np.prod(self.input_shape))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
