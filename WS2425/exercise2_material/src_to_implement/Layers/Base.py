class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None  # to be initialized in child class

    def forward(self, input_tensor):
        raise NotImplementedError("forward method not implemented in child class.")

    def backward(self, error_tensor):
        raise NotImplementedError("backward method not implemented in child class.")
