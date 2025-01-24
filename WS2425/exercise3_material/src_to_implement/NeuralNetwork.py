import numpy as np  # type: ignore
import copy
import pickle
import os
from Optimization.Loss import BaseLoss
from Optimization.Optimizers import BaseOptimizer
from Layers.Base import InitializableLayer

def save(filename, net):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(net, f)


def load(filename, data_layer):
    with open(filename, "rb") as f:
        net = pickle.load(f)
    net.data_layer = data_layer
    return net

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.data_layer = None
        self.optimizer: BaseOptimizer = optimizer
        self.loss_layer: BaseLoss = None
        self.layers = list()
        self.loss = list()
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        x, self.y = copy.deepcopy(self.data_layer.next())

        norm_loss = 0
        for layer in self.layers:
            layer.testing_phase = False
            x = layer.forward(x)
            if layer.trainable:
                if layer.optimizer.regularizer is not None:
                    norm_loss += layer.optimizer.regularizer.norm(layer.weights)

        return self.loss_layer.forward(x, copy.deepcopy(self.y)) + norm_loss

    def backward(self):
        y = self.loss_layer.backward(true_label=copy.deepcopy(self.y))
        for layer in reversed(self.layers):
            y = layer.backward(y)

    def append_layer(self, layer):
        if isinstance(layer, InitializableLayer) and layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = "train"
        for epoch in range(iterations):
            current_loss = self.forward()
            self.loss.append(current_loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = "test"
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def __rshift__(self, next_layer):
        self.append_layer(next_layer)
        return self

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
        for layer in self.layers:
            layer.set_phase(phase)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data_layer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)