class BaseLayer:
    def __init__(self):
        self.trainable = False  # False if Layers have no weights
        # self.weights = None
        self.testing_phase = False #False if training
