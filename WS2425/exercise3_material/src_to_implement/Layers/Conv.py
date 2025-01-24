import numpy as np #type: ignore
from Layers.Base import InitializableLayer, InitializableWithPhaseSeperationLayer
from Optimization.Optimizers import BaseOptimizer
from Layers.Initializers import BaseInitializer
from enum import Enum
from typing import Dict
from scipy import signal #type: ignore
from copy import deepcopy

class Conv(InitializableWithPhaseSeperationLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels) -> None:
        super().__init__()
        self.trainable = True

        if isinstance(stride_shape, int):
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        
        self.convolution_shape = convolution_shape
        self.stride_shape = stride_shape
        self.num_kernels = num_kernels

        self.IsConv2D = len(convolution_shape) == 3 
        self.weights = np.random.uniform(0, 1, size = (num_kernels, *convolution_shape))

        self.bias = np.random.uniform(0, 1, size = (num_kernels))
       
        if self.IsConv2D:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]

        self.gradient_weights  = np.zeros_like(self.weights)
        self.gradient_bias     = np.zeros_like(self.bias)

        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        if not self.IsConv2D:
            input_tensor = input_tensor[:, :, :, np.newaxis]
        
        b_size, C_in, H_in, W_in = input_tensor.shape
        _, K_h, K_w = self.convolution_shape
        S_h, S_w = self.stride_shape

        self.padded_input = self.apply_padding(input_tensor, b_size, C_in, H_in, W_in, K_h, K_w)

        b_size, C_in, H_in, W_in = self.padded_input.shape
        H_out = np.ceil((H_in - K_h + 1) / S_h).astype(int)
        W_out = np.ceil((W_in - K_w + 1) / S_w).astype(int)

        self.output_shape = (b_size, self.num_kernels, H_out, W_out)
        output_tensor = np.zeros(self.output_shape)
        
        for b_idx in range(b_size):
            for k_idx in range(self.num_kernels):
                for c_idx in range(C_in):
                    corr_output = signal.correlate(
                        self.padded_input[b_idx, c_idx],
                        self.weights[k_idx, c_idx],
                        mode = 'valid'
                    )
                    if self.IsConv2D:
                        corr_output = corr_output[::S_h, ::S_w]
                    else:
                        corr_output = corr_output[::S_h]

                    output_tensor[b_idx, k_idx] += corr_output

                output_tensor[b_idx, k_idx] += self.bias[k_idx]

        if not self.IsConv2D:
            output_tensor = output_tensor.squeeze(axis = 3)
        
        return output_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.output_shape)

        if not self.IsConv2D:
            self.input_tensor = self.input_tensor.reshape((*self.input_tensor.shape, 1))

        self.gradient_weights.fill(0)
        self.gradient_bias = np.sum(error_tensor, axis = (0, 2, 3) if self.IsConv2D else (0, 2)) # 0 -> b, 2 -> h, 3 -> w
        
        _, K_h, K_w = self.convolution_shape
        S_h, S_w = self.stride_shape
        b_size, C_in, H_in, W_in = self.input_tensor.shape
        
        padded_input_tensor = self.apply_padding(self.input_tensor, b_size, C_in, H_in, W_in, K_h, K_w)
        new_error_tensor = np.zeros_like(self.input_tensor)

        for batch_idx in range(b_size):
            for kernel_idx in range(self.num_kernels):
                for channel_idx in range(C_in):
                    upsampled_error = np.zeros((H_in, W_in))
                    
                    if self.IsConv2D:
                        upsampled_error[0::S_h, 0::S_w] = error_tensor[batch_idx, kernel_idx]
                    else:
                        upsampled_error[0::S_h] = error_tensor[batch_idx, kernel_idx]

                       
                    self.gradient_weights[kernel_idx, channel_idx] += signal.correlate(
                        padded_input_tensor[batch_idx, channel_idx], 
                        upsampled_error,
                        mode = 'valid'
                    )
                    
                    new_error_tensor[batch_idx, channel_idx] += signal.convolve(
                        upsampled_error,
                        self.weights[kernel_idx, channel_idx],
                        mode = 'same'
                    )

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        if not self.IsConv2D:
            new_error_tensor = new_error_tensor.squeeze(axis = 3) 

        return new_error_tensor

    def apply_padding(self, input_tensor, b_size, C_in, H_in, W_in, K_h, K_w):
        padded_input = np.zeros((b_size, C_in, (H_in + K_h - 1), (W_in + K_w - 1)))
        
        if K_h == 1 and K_w == 1:
            padded_input = input_tensor
        else:
            padding_height = K_h // 2
            padding_width  = K_w // 2
            is_even_K_h = K_h % 2 == 0
            is_even_K_w = K_w % 2 == 0
            padded_input[:, :, padding_height:(-padding_height+is_even_K_h), padding_width:(-padding_width+is_even_K_w)] = input_tensor
        
        return padded_input

    def initialize(self, weights_initializer: BaseInitializer, bias_initializer: BaseInitializer):
        fan_in, fan_out = np.prod(self.convolution_shape), self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = deepcopy(optimizer)
        self._optimizer.bias = deepcopy(optimizer)

