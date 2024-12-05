import numpy as np


# Here is Sgd from exe 1
class Sgd:  # S.tochastic G.radient D.escent Algorithm
    def __init__(self, learning_rate: float):
        # data type fload for learning rate
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # formular: w^(k+1) = w^(k) - n*delta_L(w(k))
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.v_k = None  # velocity
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # momentum_rate is mu = {0.9, 0.95, 0.99}, in slide.
        w_k = weight_tensor  # weight_tensor is w_k
        if self.v_k is None:
            # Return an array of zeros with the same shape and type as a given array.
            self.v_k = np.zeros_like(w_k)
        self.v_k = self.momentum_rate * self.v_k - self.learning_rate * gradient_tensor
        # update w_k_1 = w_k + v_k
        updated_weight = weight_tensor + self.v_k
        return updated_weight


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.v = None
        self.r = None
        self.epsilon = np.finfo(np.float64).eps
        self.k = 1
        self.learning_rate = learning_rate
        self.mu = mu  # u
        self.rho = rho  # p

    def calculate_update(self, weight_tensor, gradient_tensor):
        g = gradient_tensor
        if self.v is None:
            # Return an array of zeros with the same shape and type as a given array.
            self.v = np.zeros_like(weight_tensor)
        if self.r is None:
            # Return an array of zeros with the same shape and type as a given array.
            self.r = np.zeros_like(gradient_tensor)
        # parameter update based on current and past gradients
        self.v = self.mu * self.v + (1 - self.mu) * g
        self.r = self.rho * self.r + (1 - self.rho) * (g**2)
        # bias correction
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        # w_k+1 = w_k - learning_rate*(v_hat / (sqrt(r_hat)+epsilon))
        updated_weight = weight_tensor - self.learning_rate * (v_hat) / (
            np.sqrt(r_hat) + self.epsilon
        )
        self.k = self.k + 1
        return updated_weight
