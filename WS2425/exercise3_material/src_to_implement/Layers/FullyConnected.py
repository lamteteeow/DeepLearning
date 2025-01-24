import numpy as np
import random
from Optimization.Optimizers import Sgd
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__() #super constructor
        self.trainable = True # set member to True, as this layer has trainable parameters
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None # private/protected member
        self.weights = np.random.uniform(low= 0, high = 1, size = ( input_size + 1, output_size)) 

    def initialize(self, weights_init, bias_init):
        weights = weights_init.initialize(np.shape(self.weights[:-1, :]), np.shape(self.weights[:-1, :])[0],
                                          np.shape(self.weights[:-1, :])[1])

        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_init.initialize(bias.shape, bias.shape[0], bias.shape[1])

        self.weights = np.concatenate((weights, bias))
        
    def forward(self,input_tensor):
        input_tensor = np.append(input_tensor,np.ones((input_tensor.shape[0],1 )), axis = 1) 
        self.input_tensor = input_tensor # needed for backward

        #Calculate input tensor for next layer
        next_input_tensor = input_tensor @ self.weights 
        
        return next_input_tensor
    
    def backward(self, error_tensor):
        # Calculate gradient with respect to weights
        self._gradient_weights = self.input_tensor.T @ error_tensor

        # Check if optimizer is set, if yes, then update weight
        if self.optimizer is None: 
            pass
        else:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        
        # Calculate error tensor for previous layer
        prev_error_tensor =  error_tensor @ self.weights.T[:,:-1] 
        # print(f"{self.input_tensor.T.shape} {error_tensor.shape} {self.weights.shape} {self.gradient_weights.shape}")
        return  prev_error_tensor
    
    @property
    def optimizer(self): #getter
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, new_value): #setter new value
        self._optimizer = new_value

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    