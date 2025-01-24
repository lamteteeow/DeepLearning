import numpy as np #type: ignore
from abc import ABC, abstractmethod


class BaseInitializer(ABC):
    @abstractmethod
    def initialize(self, shape, fan_in, fan_out):
        pass

class Constant(BaseInitializer):
    def __init__(self, value:float = 0.1) -> None:
        super().__init__()
        self.value = value

    def initialize(self, shape, fan_in, fan_out):
        return np.zeros(shape) + self.value
    

class UniformRandom(BaseInitializer):
    def initialize(self, shape, fan_in, fan_out):
        return np.random.uniform(0, 1, size = shape)
    
class Xavier(BaseInitializer):
    def initialize(self, shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, sigma, size = shape)
    
class He(BaseInitializer):
    def initialize(self, shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, sigma, size = shape)