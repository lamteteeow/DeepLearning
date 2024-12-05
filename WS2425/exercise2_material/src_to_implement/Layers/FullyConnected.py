import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()  # super constructor
        self.trainable = True  # set member to True, as this layer has trainable parameters
        self.input_size = input_size
        self.output_size = output_size

        self._optimizer = None  # private/protected member

        self.weights = np.random.uniform(
            low=0, high=1, size=(input_size + 1, output_size)
        )  # for n input neurons and m output neurons, weight matrix would be (n x m)

    def initialize(self, weights_init, bias_init):
        weights = weights_init.initialize(
            np.shape(self.weights[:-1, :]),
            np.shape(self.weights[:-1, :])[0],
            np.shape(self.weights[:-1, :])[1],
        )

        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_init.initialize(bias.shape, bias.shape[0], bias.shape[1])

        self.weights = np.concatenate((weights, bias))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  # batch size x input size
        # Initialize input tensor:   input tensor = batch size x input size
        # adding 1 - create a bias to add to weight matrix
        bias = np.ones((input_tensor.shape[0], 1))
        # self.input_tensor_with_bias = np.append(input_tensor,bias, axis = 1) #Array of ones with batch_size x 1
        # calculate input_tensor_with_bias - at the end of each input_tensor array, adding 1: slide 8
        self.input_tensor_with_bias = np.hstack((input_tensor, bias))
        # Calculate input tensor for next layer
        next_input_tensor = np.dot(
            self.input_tensor_with_bias, self.weights
        )  # (batch size x input size) x (input size x output size) = (batch size x output size)
        # y_hat_estimated = input(n+1) = weight_n.input_n
        return next_input_tensor

    def backward(self, error_tensor):
        weight_without_bias = self.weights.T[:, :-1]  # En−1 = W_T.En ## remove bias
        # Calculate gradient with respect to weights:    η · En.XT
        self._gradient_weights = np.dot(
            self.input_tensor_with_bias.T, error_tensor
        )  # gradient_weights = (input x output)
        # Check if optimizer is set, then use method calculate_update(weight_tensor, gradient_tensor):
        if self.optimizer is None:
            pass
        else:  # update W using gradient with respect to W
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        # self_weights = (input size + 1) x output size
        # error_tensor = weight x output size
        # prev_error_tensor =  batch x input_size

        # Calculate error tensor for previous layer:
        # shape(error_tensor) = batch_size x output_size
        # shape(weights) = input x output
        # error tensor for previous layer has a shape of (batch x input)
        # En−1 = W_T.En
        prev_error_tensor = np.dot(error_tensor, weight_without_bias)
        return prev_error_tensor

    @property
    def optimizer(self):  # getter
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_value):  # setter new value
        self._optimizer = new_value

    @property
    def gradient_weights(self):
        return self._gradient_weights
