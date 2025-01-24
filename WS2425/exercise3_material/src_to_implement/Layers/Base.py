from abc import ABC, abstractmethod
from .Initializers import BaseInitializer

class BaseLayer(object):
    def __init__(self):
        self.trainable = False
        self.weights = []
        self.testing_phase = False

class InitializableLayer(BaseLayer):
    def initialize(self, weights_initializer: BaseInitializer, bias_initializer: BaseInitializer):
        pass

class PhaseSeperatableLayer(BaseLayer):
    def set_phase(self, phase):
        self.testing_phase = phase

class InitializableWithPhaseSeperationLayer(InitializableLayer):
    def set_phase(self, phase):
        self.testing_phase = phase