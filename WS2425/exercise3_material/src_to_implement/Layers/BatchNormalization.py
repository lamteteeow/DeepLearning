import numpy as np #type: ignore
import pickle
from copy import deepcopy
from Layers.Base import (
    BaseInitializer,
    InitializableWithPhaseSeperationLayer
)

from Layers.Helpers import compute_bn_gradients

class BatchNormalization(InitializableWithPhaseSeperationLayer):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self._optimizer  = None
        self.running_mean = None
        self.running_var = None
        self.alpha = 0.8
        self.trainable = True 
        self.initialize(None, None)

    def initialize(self, weights_initializer: BaseInitializer, bias_initializer: BaseInitializer):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            self.corrected_shape = tensor.shape
            _, C, _, _ = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            B, C, H, W = self.corrected_shape
            return tensor.reshape(B, H * W, C).transpose(0, 2, 1).reshape(B, C, H, W)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_conv = input_tensor.ndim == 4
        
        if is_conv:
            self.input_tensor = self.reformat(self.input_tensor)
        
        if self.testing_phase:
            self.batch_mean = self.running_mean
            self.batch_var = self.running_var
        else:
            self.batch_mean = np.mean(self.input_tensor, axis= 0)
            self.batch_var = np.var(self.input_tensor, axis=0)
            if self.running_mean is None:
                self.running_mean = deepcopy(self.batch_mean)
                self.running_var = deepcopy(self.batch_var)
            else:
                self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.batch_mean 
                self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.batch_var
        
        self.X_tilda = (self.input_tensor - self.batch_mean) / np.sqrt(self.batch_var + np.finfo(float).eps)
        output_tensor = self.gamma * self.X_tilda + self.beta
        
        if is_conv:
            output_tensor = self.reformat(output_tensor)
            
        return output_tensor
    
    def backward(self, error_tensor):
        is_conv = error_tensor.ndim == 4
        
        if is_conv:
            error_tensor = self.reformat(error_tensor)

        self.gradient_weights = np.sum(error_tensor * self.X_tilda, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        
        input_grad = compute_bn_gradients(
            error_tensor, 
            self.input_tensor, 
            self.gamma, 
            self.batch_mean, 
            self.batch_var
        )
        
        if self._optimizer is not None:
            self.weights = self._optimizer.weight.calculate_update(self.gamma, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.beta, self.gradient_bias)

        if is_conv:
            input_grad = self.reformat(input_grad)
        
        return input_grad
    
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    def set_weight(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = deepcopy(optimizer)
        self._optimizer.bias = deepcopy(optimizer)

