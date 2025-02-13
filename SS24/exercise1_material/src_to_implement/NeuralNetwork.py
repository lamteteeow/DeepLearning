import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        # self.data_layer = None
        # self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        # Provide 2 variables by calling next() from data layer:
        input_tensor, label_tensor = self.data_layer.next()  # from Helpers.py line 161

        # Initialize for backward:
        self.label_tensor = label_tensor

        # Forward input_tensor through the whole network (of Layers and Optimizers folder):
        # need to call this loop recursively
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, label_tensor)
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

        # Trying to make it recursive:
        # loss = self.forward()
        # self.loss.append(loss)
        # self.backward()
        # if iterations > 0:
        #     self.train(iterations - 1)

    def test(self, input_tensor):
        # need to call this loop recursively
        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor
