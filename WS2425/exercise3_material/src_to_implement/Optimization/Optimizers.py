import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate 

    def calculate_update(self, weight_tensor, gradient_tensor):
        #Check if the new regularizer is set:
        
        if self.regularizer is None:
            updated_weight = weight_tensor - self.learning_rate *  gradient_tensor #Updating weights without regularization
        else:
            gradient_regularizer = self.regularizer.calculate_gradient(weight_tensor)
            updated_weight = weight_tensor - self.learning_rate * (gradient_regularizer + gradient_tensor) #with regularization -> including the gradient of norm
        return updated_weight

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.v_k = None     # velocity
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Momentum_rate is mu = {0.9, 0.95, 0.99}, in slide.
        
        if self.v_k is None:
            # Return an array of zeros with the same shape and type as a given array.
            self.v_k = np.zeros_like(weight_tensor)
        
        # Check if the new regularizer is set:
        
        if self.regularizer is None:
            pass
        else: 
            gradient_regularizer = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate* gradient_regularizer # Weight tensor with regularization
        self.v_k = self.momentum_rate*self.v_k - self.learning_rate*(gradient_tensor) #Gradient vk
        updated_weight = weight_tensor + self.v_k
        return updated_weight

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.v = None
        self.r = None
        self.epsilon = np.finfo(np.float64).eps
        self.k = 1
        self.learning_rate = learning_rate
        self.mu = mu # u
        self.rho = rho # p

    def calculate_update(self, weight_tensor, gradient_tensor):
        
        #Check if regularizer is set, gradient regularizer is included: 
        if self.regularizer is None:
            pass 
        else: 
            gradient_regularizer = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * gradient_regularizer
        
        if self.v is None:
            # Return an array of zeros with the same shape and type as a given array.
            self.v = np.zeros_like(weight_tensor)
            
        if self.r is None:
            # Return an array of zeros with the same shape and type as a given array.
            self.r = np.zeros_like(gradient_tensor)
        # parameter update based on current and past gradients
        self.v = self.mu*self.v + (1-self.mu)*gradient_tensor
        self.r = self.rho*self.r + (1-self.rho)*(gradient_tensor**2)
        # bias correction
        v_hat = self.v / (1-np.power(self.mu, self.k))
        r_hat = self.r / (1-np.power(self.rho, self.k))
        # w_k+1 = w_k - learning_rate*(v_hat / (sqrt(r_hat)+epsilon))
        updated_weight = weight_tensor - self.learning_rate*(v_hat) / (np.sqrt(r_hat)+self.epsilon)
        self.k = self.k + 1
        return updated_weight
    