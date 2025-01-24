from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None
    def forward(self, input_tensor):
        # Randomly set activations to 0 with probability 1-p
        self.mask = np.random.rand(*input_tensor.shape) < self.probability # p of number 1, 1-p of number 0
        self.mask = np.where(self.mask, 1, 0) # Replace True with 1, False with 0
        if self.testing_phase:
            return input_tensor
        else: # Training phase           
            # Inverted dropout
            output = input_tensor * self.mask / self.probability
            return output
    
    def backward(self, error_tensor): 
        pre_error_tensor = error_tensor * self.mask / self.probability
        return pre_error_tensor