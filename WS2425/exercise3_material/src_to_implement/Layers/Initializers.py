import numpy as np

# Each of them has to provide the method initialize(weights shape,
# fan in, fan out) which returns an initialized tensor of the desired shape
class Constant:
    def __init__(self, weight_constant=0.1):
        # a member that determines the constant value used for weight initialization
        self.weight_constant = weight_constant
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        #weights_shape = (fan_in, fan_out)   ##############
        #passed with a default of 0.1 value
        #Return a new array of given shape and type, filled with fill_value
        self.weights = np.full(weights_shape, self.weight_constant)
        return self.weights

class UniformRandom:
    def __init__(self):
        self.weights = None#

    def initialize(self, weights_shape, fan_in, fan_out):
        # weights_shape = (fan_in, fan_out)    ##############
        # with the size of weight_shape
        self.weights = np.random.uniform(0, 1, weights_shape)
        return self.weights

class Xavier:
    def __init__(self):
        self.weights = None#

    def initialize(self, weights_shape, fan_in, fan_out):
        #formula of Zero-mean Gaussian N(0, sigma)
        sigma = np.sqrt(2/(fan_in + fan_out))
        self.weights = np.random.normal(0, sigma, weights_shape)
        return self.weights

class He:
    def __init__(self):
        self.weights = None#

    def initialize(self, weights_shape, fan_in, fan_out):
        #weights_shape = (fan_in, fan_out)    ##############
        sigma = np.sqrt(2 / fan_in)
        self.weights = np.random.normal(0, sigma, weights_shape)
        return self.weights
