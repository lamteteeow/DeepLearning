from Layers.Base import BaseLayer
import numpy as np
from Layers.Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True #layers have weights
        self.channels = channels
        self.bias =  None # Beta
        self.weights = None # Gramma
        self._optimizer = None # private/protected member

        self.epsilon = 1e-10
        self.running_mean = None
        self.running_var = None
        self.gradient_bias = None
        self.gradient_weights = None
    
    def initialize(self, weights_init, bias_init):
        self.weights = np.ones((self.channels, ))
        self.bias = np.zeros((self.channels, ))

    def reformat(self, input_tensor):
        if len(input_tensor.shape) == 4:
            b, h, m, n = input_tensor.shape
            # Reshape B x H x M x N -> B x H x M*N
            reshaped_vec = input_tensor.reshape(b,h, m*n)
            # Transpose B x H x M*N -> B x M*N x H 
            transpose_vec = np.transpose(reshaped_vec, (0, 2, 1))
            # Reshape B x H*M x N -> B*H*M x N
            new_input_tensor = transpose_vec.reshape(transpose_vec.shape[0]*transpose_vec.shape[1], transpose_vec.shape[2])
            return new_input_tensor
        elif len(input_tensor.shape) == 2:
            b, h, m, n = self.initial_shape # shape of the original 4d array
            # Reshape B*M*N x H to B x M*N x H
            reshaped_vec = input_tensor.reshape(b, m*n, h)
            # Transpose B x H x M*N
            transpose_vec = np.transpose(reshaped_vec, (0,2,1))
            # Reshape
            new_input_tensor = transpose_vec.reshape(b,h,m,n)
            return new_input_tensor     

    def forward(self, input_tensor):
        # Reformat the input tensor from  dim 4 to dim 2
        self.initial_shape = input_tensor.shape # Save the initial input shape for backward
        if len(input_tensor.shape) == 4: 
            self.input_tensor = self.reformat(input_tensor)
        else: 
            self.input_tensor = input_tensor

        eps= np.finfo(float).eps
        
        # Initialize gamma and beta when they are None
        if self.weights is None and self.bias is None:
            self.initialize(None, None) # because the weights and bias initializers are ignored, we can just use None here
        
        # Check the criteria
        if eps > 1e-10:
            raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" %(str(eps)))
        
        # Compute mean and variance of batch
        self.batch_mean = np.mean(self.input_tensor, axis = 0, keepdims = True) # Mean of batch
        self.batch_var = np.var(self.input_tensor, axis = 0, keepdims = True) # Variance of batch = Std **2
        
        # Forward calculation
        if self.testing_phase == False: 
            # Training phase
            # Online estimation of mean
            if self.running_mean is None:
                self.running_mean = self.batch_mean
            else: 
                self.running_mean = 0.8 * self.running_mean + 0.2 * self.batch_mean

            # Online estimation of variance:
            if self.running_var is None:
                self.running_var = self.batch_var
            else: 
                self.running_var = 0.8 * self.running_var + 0.2 * self.batch_var
            # Normalization for training
            self.X_tilde = (self.input_tensor - self.batch_mean) / np.sqrt(self.batch_var + eps)          
        else: 
            # Testing phase         
            # Normalization for testing
            self.X_tilde = (self.input_tensor - self.running_mean) / np.sqrt(self.running_var + eps)
           
        
        # Calculate output
        output_tensor = self.weights * self.X_tilde + self.bias
        
        # Reformat the output from  dim 2 to dim 4
        if len(self.initial_shape) == 4: # initial input shape
            output_tensor = self.reformat(output_tensor)
        return output_tensor
    
    def backward(self, error_tensor):
        # Check the criteria
        eps= np.finfo(float).eps
        if eps > 1e-10:
            raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" %(str(eps)))
        
        # Reformat the error tensor if CNN
        
        if len(error_tensor.shape) == 4: 
           self.initial_shape = error_tensor.shape
           error_tensor = self.reformat(error_tensor) # reform it to 2 dimensions
        
        # Calculate gradient
        self.gradient_weights = np.sum(error_tensor* self.X_tilde, axis = 0 ) # Sum of all batches
        self.gradient_bias = np.sum(error_tensor, axis = 0 ) # Sum of all batches

        # Check if optimizer are defined
        if self.optimizer is not None:
            # Gamma: Weight
            updated_gamma = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.weights = updated_gamma # Save the updated gamme into gamma
            
            # Beta: Bias
            updated_beta = self.optimizer.calculate_update(self.bias, self.gradient_bias)
            self.bias = updated_beta # Save the updated gamme into beta
        
        # Error tensor respect to input = Previous error tensor
        prev_error_tensor = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.batch_mean, self.batch_var, eps=np.finfo(float).eps)
        
        # Reform the previous error tensor if the initial dim is 4
        if len(self.initial_shape) == 4: # Check the original shape before the first reformat in Line 101
            prev_error_tensor = self.reformat(prev_error_tensor) # reform it back to 4 dimensions

        return prev_error_tensor
    @property
    def optimizer(self): #getter
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, new_value): #setter new value
        self._optimizer = new_value


