import copy
import numpy as np
from Layers import *
from Optimization import *

class NeuralNetwork:
    def __init__(self, optimizer, weightsInitializer, biasInitializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weightsInitializer = weightsInitializer
        self.biasInitializer = biasInitializer
        # Use for property "Phase"
        self._phase = "train"

    def forward(self):
         #Provide 2 variables by calling next() from data layer:
        input_tensor , label_tensor = self.data_layer.next() #from Helpers.py line 161

        #Initialize for backward:
        self.label_tensor = label_tensor
        
        #Forward input_tensor through the whole network (of Layers and Optimizers folder):
        tensor = input_tensor
        # Sum the regularization loss up
        norm_loss = 0
        for layer in self.layers:
            tensor = layer.forward(tensor)
            if layer.trainable == True: 
                if layer.optimizer.regularizer is not None:
                    regularization_loss= layer.optimizer.regularizer.norm(layer.weights)
                    norm_loss += regularization_loss
        loss = self.loss_layer.forward(tensor, label_tensor) + norm_loss

        return loss
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            # layer.initialize(copy.deepcopy(self.weightsInitializer), copy.deepcopy(self.biasInitializer))
            # layer.optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weightsInitializer, self.biasInitializer)
            
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = "train"
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = "test"
        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor

    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, new_phase):
        self._phase = new_phase
        if self._phase == "train":
            for layer in self.layers:
                layer.testing_phase = False
        elif self._phase == "test":
            for layer in self.layers:
                layer.testing_phase = True