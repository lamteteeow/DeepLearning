import numpy as np #type: ignore
from abc import ABC, abstractmethod
from copy import deepcopy


class BaseOptimizer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.regularizer = None

    @abstractmethod
    def calculate_update(self, weight_tensor, gradient_tensor): 
        pass

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def apply_regularization(self, intermediate_update, weight_tensor):
        regularization_gradient = 0      
        if self.regularizer is not None:
            regularization_gradient = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return intermediate_update - regularization_gradient

class Sgd(BaseOptimizer):
    def __init__(self, learning_rate:float):
        super().__init__()
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        intermediate_update = weight_tensor - self.learning_rate * gradient_tensor
        return self.apply_regularization(intermediate_update, weight_tensor)

class SgdWithMomentum(BaseOptimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        
        self.v = self.momentum_rate * self.v + self.learning_rate * gradient_tensor
        intermediate_update = weight_tensor - self.v
        return self.apply_regularization(intermediate_update, weight_tensor)


class Adam(BaseOptimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        if self.r is None:
            self.r = np.zeros_like(weight_tensor)

        self.k += 1

        self.v = self.mu * self.v + (1.0 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor ** 2)
        
        v_hat = self.v / (1.0 - self.mu ** self.k)
        r_hat = self.r / (1.0 - self.rho ** self.k)
        
        intermediate_update = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        return self.apply_regularization(intermediate_update, weight_tensor)
