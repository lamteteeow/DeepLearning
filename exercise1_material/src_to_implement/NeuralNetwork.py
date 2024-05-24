import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        # self.data_layer = None
        # self.loss_layer = None

    def forward(self):
        # Provide 2 variables by calling next() from data layer:
        input_tensor, label_tensor = self.data_layer.next()  # from Helpers.py line 161

        # Initialize for backward:
        self.label_tensor = label_tensor

        # Forward input_tensor through the whole network (of Layers and Optimizers folder):
        tensor = input_tensor
        # need to flatten this loop
        for layer in self.layers:
            tensor = layer.forward(tensor)
        loss = self.loss_layer.forward(tensor, label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        # need to flatten this loop
        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor
