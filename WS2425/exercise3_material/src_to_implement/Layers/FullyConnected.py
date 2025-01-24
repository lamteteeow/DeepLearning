import numpy as np # type: ignore
from Layers.Base import InitializableWithPhaseSeperationLayer
from Layers.Initializers import BaseInitializer
from Optimization.Optimizers import BaseOptimizer

class FullyConnected(InitializableWithPhaseSeperationLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.trainable  = True
        self.weights    = np.random.uniform(0, 1, (input_size + 1, output_size)) # [5 x 3]
        
        self._optimizer: BaseOptimizer  = None
        self.delta_w                    = None

    def forward(self, input_tensor):
        initial_bias = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.concatenate((input_tensor, initial_bias), axis = 1) # [9 x 5]
        return np.dot(self.input_tensor, self.weights)  # [9 x 5] (.) [5 x 3] = [9 x 3] 
    
    def backward(self, error_tensor):
        # error_tensor -> [9 x 3]
        weights = self.weights[:self.weights.shape[0]-1, :] # [4 x 3]: bias excluded
        
        # delta_a^(L-1) = w^L * activation_fn(z^L) * 2(a^L - y) 
        # w^L == weights && activation_fn(z^L) * 2(a^L - y) == error_tensor 
        next_layer_gradients = np.dot(error_tensor, weights.T)    # [9 x 4]

        # delta_w = a^(L-1) * activation_fn(z^L) * 2(a^L - y)
        # a^(L-1) == self.input_tensor && activation_fn(z^L) * 2(a^L - y) == error_tensor
        self.delta_w = np.dot(self.input_tensor.T, error_tensor) # [5 x 3] 

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(
                weight_tensor   = self.weights, 
                gradient_tensor = self.delta_w
            ) # [5 x 3]

        return next_layer_gradients
    
    def initialize(self, weights_initializer: BaseInitializer, bias_initializer: BaseInitializer):
        self.weights = weights_initializer.initialize(
            shape   = (self.input_size, self.output_size),
            fan_in  = self.input_size,
            fan_out = self.output_size
        )

        self.bias = bias_initializer.initialize(
            shape   = (1, self.output_size),
            fan_in  = 1,
            fan_out = self.output_size
        )

        self.weights = np.vstack((self.weights, self.bias))

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def gradient_weights(self):
        return self.delta_w

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def set_weight(self, weight):
        self._weights = weight