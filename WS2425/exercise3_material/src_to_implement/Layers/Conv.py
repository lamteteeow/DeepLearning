import numpy as np
import copy
from Layers.Base import BaseLayer
from scipy import signal
from scipy.signal import correlate, convolve

class Conv(BaseLayer):  # inherited member "trainable"
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True    
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)
        self.convolution_shape = convolution_shape          # Shape either 1-D -> [c,m]; m = kernel length or 2-D -> [c,m,n]
        self.num_kernels = num_kernels      # int value -> equivalent to num of outputs

        # Introduce members
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self._bias_optimizer = None
        
    def forward(self, input_tensor):
        
        # Store the input tensor for use in backward pass
        self.input_tensor = input_tensor

        # Check the shape of input tensor and return the forward accordingly
        if len(input_tensor.shape) == 3:  # 1D convolution
            return self._forward_1d(input_tensor)
        elif len(input_tensor.shape) == 4:  # 2D convolution
            return self._forward_2d(input_tensor)
        else:
            raise ValueError("Input tensor must be 3D or 4D for 1D or 2D convolution respectively.")
    
    def _forward_1d(self, input_tensor):
        # for 1D the input_tensor.shape = b, c, y
        batch_size, input_channels, input_length = input_tensor.shape
        kernel_length = self.convolution_shape[1]

        # Calculate padding for 1D
        padding_length = (kernel_length - 1) // 2

        # Save for backward
        self.padding_length_before = padding_length
        self.padding_length_after = padding_length if kernel_length % 2 == 1 else padding_length + 1

        # Add padding to input tensor
        padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (padding_length, padding_length)), mode='constant')
        
        # Add padding to output tensor
        output_length = (input_length + 2 * padding_length - kernel_length) // self.stride_shape[0][0] + 1
        output_tensor = np.zeros((batch_size, self.num_kernels, output_length))
        
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for i in range(output_length):
                    start = i * self.stride_shape[0][0]
                    end = start + kernel_length
                    output_tensor[b, k, i] = np.sum(padded_input[b, :, start:end] * self.weights[k, :, :]) + self.bias[k]

        return output_tensor

    def _forward_2d(self, input_tensor):
        # for 2D the input_tensor.shape = b, c, y, x
        batch_size, input_channels, input_height, input_width = input_tensor.shape
        kernel_height, kernel_width = self.convolution_shape[1], self.convolution_shape[2]

        # Calculate padding
            # Check if the kernel size is even number:
        padding_height_input = (kernel_height - 1) // 2
        padding_width_input = (kernel_width - 1) // 2

        if kernel_height % 2 == 0:
            padding_height_output = (kernel_height - 1) // 2 +1
        else: 
            padding_height_output = padding_height_input

        if kernel_width % 2 == 0:
            padding_width_output = (kernel_width - 1) // 2 + 1
        else: 
            padding_width_output = padding_width_input

        # Store padding heights and widths for backward convenience
        self.padding_height_before = padding_height_input
        self.padding_height_after = padding_height_output
        self.padding_width_before = padding_width_input
        self.padding_width_after = padding_width_output

        # Add padding to input tensor
        padded_input = np.pad(input_tensor, 
                            ((0, 0), (0, 0), (padding_height_input, padding_height_output), (padding_width_input, padding_width_output)), 
                            mode='constant', constant_values=0)

        # Calculate output dimensions and add padding to error tensor
        output_height = (input_height + padding_height_input + padding_height_output - kernel_height) // self.stride_shape[0] + 1
        output_width = (input_width +  padding_width_input + padding_width_output - kernel_width) // self.stride_shape[1] + 1

        # Initialize the output tensor
        output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

        # Perform the correlation (which is essentially convolution without kernel flipping)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                # Initialize the sum for this kernel
                summed_output = np.zeros((output_height, output_width))
                for c in range(input_channels):
                    correlation_result = correlate(padded_input[b, c], self.weights[k, c], mode='valid')
                    # Ensure the dimensions are correct by applying stride
                    sliced_result = correlation_result[::self.stride_shape[0], ::self.stride_shape[1]]
                    # Sum the results across channels
                    summed_output += sliced_result
                # Add the bias term5
                output_tensor[b, k] = summed_output + self.bias[k]
        return output_tensor    

    def backward(self, error_tensor):
        """Backward pass to compute gradients."""
        if len(self.input_tensor.shape) == 3:  # 1D convolution
            return self._backward_1d(error_tensor)
        elif len(self.input_tensor.shape) == 4:  # 2D convolution
            return self._backward_2d(error_tensor)
        else:
            raise ValueError("Input tensor must be 3D or 4D for 1D or 2D convolution respectively.")
        
        # "same" convolution across the image plane axes

    def _backward_1d(self, error_tensor):
        # Almost copy paste from 2d case with small modifications
        prev_error_tensor = np.zeros_like(self.input_tensor) # Error for lower layer
        batch_size, input_channels, input_length = self.input_tensor.shape
        _, kernel_length = self.convolution_shape

        # Upsampling input (check slides)
        padded_input = np.pad(
            self.input_tensor, 
            ((0, 0), (0, 0), (self.padding_length_before, self.padding_length_after)), 
            mode='constant',
            constant_values=0,
        )

        # Upsample error_tensor due to striding
        # Upsampled length comes from result of correlate(padded_input, kernel)
        upsampled_length = padded_input.shape[2] - kernel_length + 1
        upsampled = np.zeros((batch_size, self.num_kernels, upsampled_length))
        upsampled[:, :, ::self.stride_shape[0][0]] = error_tensor[:, :]
        error_tensor = upsampled

        # First compute the error signal (gradient w.r.t. input)
        for b in range(batch_size):
            # Construct #input_channels kernels with depth #num_kernels
            new_kernels = np.transpose(self.weights, (1, 0 , 2))
            for c in range(input_channels):
                error_channel_c = np.zeros((input_length,))
                for k in range(self.num_kernels):
                    error_k = convolve(error_tensor[b, k], new_kernels[c, k], mode="same")
                    error_channel_c += error_k
                prev_error_tensor[b, c] = error_channel_c

        # Now compute gradient w.r.t. weights
        grad_weights = np.zeros_like(self.weights)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(input_channels):
                    grad_weights[k, c] += correlate(padded_input[b, c], error_tensor[b, k], mode="valid")
        self.gradient_weights = grad_weights
        self.gradient_bias = error_tensor.sum(axis=(0, 2))

        # Update weights and bias if optimizer available
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)
        return prev_error_tensor

    def _backward_2d(self, error_tensor):
        prev_error_tensor = np.zeros_like(self.input_tensor) # Error for lower layer
        batch_size, input_channels, input_height, input_width = self.input_tensor.shape
        _, kernel_height, kernel_width = self.convolution_shape

        # Upsampling input (check slides)
        padded_input = np.pad(
            self.input_tensor, 
            ((0, 0), (0, 0), (self.padding_height_before, self.padding_height_after), (self.padding_width_before, self.padding_width_after)), 
            mode='constant',
            constant_values=0,
        )

        # Upsample error_tensor due to striding
        # Upsampled height and width comes from result of correlate(padded_input, kernel)
        upsampled_height = padded_input.shape[2] - kernel_height + 1
        upsampled_width = padded_input.shape[3] - kernel_width + 1
        upsampled = np.zeros((batch_size, self.num_kernels, upsampled_height, upsampled_width))
        upsampled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[:, :]
        error_tensor = upsampled

        # First compute the error signal (gradient w.r.t. input)
        for b in range(batch_size):
            # Construct #input_channels kernels with depth #num_kernels
            new_kernels = np.transpose(self.weights, (1, 0 , 2, 3))
            for c in range(input_channels):
                error_channel_c = np.zeros((input_height, input_width))
                for k in range(self.num_kernels):
                    error_k = convolve(error_tensor[b, k], new_kernels[c, k], mode="same")
                    error_channel_c += error_k
                prev_error_tensor[b, c] = error_channel_c
        

        # Now compute gradient w.r.t. weights
        grad_weights = np.zeros_like(self.weights)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(input_channels):
                    grad_weights[k, c] += correlate(padded_input[b, c], error_tensor[b, k], mode="valid")
        self.gradient_weights = grad_weights
        self.gradient_bias = error_tensor.sum(axis=(0, 2, 3))

        # Update weights and bias if optimizer available
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)
        return prev_error_tensor


    def initialize(self, weights_initializer, bias_initializer):
        # Calculate fan_in and fan_out, check the dimension for 1D and 2D
        fan_in = self.convolution_shape[0] * np.prod(self.convolution_shape[1:]) 
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:]) 
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

    def _calculate_output_shape(self, input_tensor):
        if len(input_tensor.shape) == 3:
            batch_size, channels, length = input_tensor.shape
            output_length = (length - self.convolution_shape[1]) // self.stride_shape[0] + 1
            return (batch_size, self.num_kernels, output_length)
        elif len(input_tensor.shape) == 4:
            batch_size, channels, height, width = input_tensor.shape
            output_height = (height - self.convolution_shape[1]) // self.stride_shape[0] + 1
            output_width = (width - self.convolution_shape[2]) // self.stride_shape[1] + 1
            return (batch_size, self.num_kernels, output_height, output_width)
        else:
            raise ValueError("Input tensor must be 3D or 4D for 1D or 2D convolution respectively.")

    ###Properties

    @property
    def optimizer(self): #getter
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, new_value): #setter new value
        # self._optimizer = copy.deepcopy(new_value)
        self._optimizer = new_value
        # self.weightsOptimizer = copy.deepcopy(new_value)
        # self.biasOptimizer = copy.deepcopy(new_value)

    @property
    def gradient_weights(self):#getter
        return self._gradient_weights
    @gradient_weights.setter#setter new value
    def gradient_weights(self, value):
        self._gradient_weights = value
    
    @property
    def gradient_bias(self):#getter
        return self._gradient_bias
    @gradient_bias.setter#setter new value
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def bias_optimizer(self):#getter
        return self._bias_optimizer
    @bias_optimizer.setter#setter new value
    def bias_optimizer(self, value):
        self._bias_optimizer = value