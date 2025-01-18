from Layers.Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()  # super constructor
        self.stride_shape = (
            stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)
        )
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_shape = self._calculate_output_shape(input_tensor)
        output_tensor = np.zeros(output_shape)
        self.mask = np.zeros(
            (
                input_tensor.shape[0],
                input_tensor.shape[1],
                output_shape[2],
                output_shape[3],
                self.pooling_shape[0],
                self.pooling_shape[1],
            )
        )

        for i in range(output_shape[2]):
            for j in range(output_shape[3]):
                region = input_tensor[
                    :,
                    :,
                    i * self.stride_shape[0] : i * self.stride_shape[0] + self.pooling_shape[0],
                    j * self.stride_shape[1] : j * self.stride_shape[1] + self.pooling_shape[1],
                ]
                output_tensor[:, :, i, j] = np.max(region, axis=(2, 3))

                # Create mask for backpropagation
                # Each output pixel corresponds to 1 region of input
                max_region = region == np.max(region, axis=(2, 3), keepdims=True)
                self.mask[:, :, i, j] = max_region

        return output_tensor

    def backward(self, error_tensor):
        input_grad = np.zeros_like(self.input_tensor)

        for i in range(error_tensor.shape[2]):
            for j in range(error_tensor.shape[3]):
                grad = error_tensor[:, :, i, j][:, :, None, None]  # (B, C, 1, 1)
                input_grad[
                    :,
                    :,
                    i * self.stride_shape[0] : i * self.stride_shape[0] + self.pooling_shape[0],
                    j * self.stride_shape[1] : j * self.stride_shape[1] + self.pooling_shape[1],
                ] += grad * self.mask[:, :, i, j]
        return input_grad

    def _calculate_output_shape(self, input_tensor):
        batch_size, channels, height, width = input_tensor.shape
        output_height = (height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        output_width = (width - self.pooling_shape[1]) // self.stride_shape[1] + 1
        return (batch_size, channels, output_height, output_width)
