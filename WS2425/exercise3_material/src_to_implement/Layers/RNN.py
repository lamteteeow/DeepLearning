import numpy as np
from copy import deepcopy
from Layers.Base import InitializableWithPhaseSeperationLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

XT_TILDA_ACTIVATION = 'XT_TILDA_ACTIVATION'
YT_ACTIVATION       = 'YT_ACTIVATION'
TANH_ACTIVATION     = 'TANH_ACTIVATION'
SIGMOID_ACTIVATION  = 'SIGMOID_ACTIVATION'

class RNN(InitializableWithPhaseSeperationLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable   = True
        self._memorize   = False
        self.regularizer = None

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_xt = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_yt = FullyConnected(hidden_size, output_size)

        self.previous_ht      = np.zeros(self.hidden_size)
        self.weights = self.fc_xt.weights
        
        self.tanh_layer    = TanH()
        self.sigmoid_layer = Sigmoid()        
        self._optimizer    = None
        self._optimizer_yt = None

        self.state = {
            XT_TILDA_ACTIVATION : {},
            YT_ACTIVATION       : {},
            TANH_ACTIVATION     : {},
            SIGMOID_ACTIVATION  : {}
        }

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size   = input_tensor.shape[0]

        self.hts = np.zeros((self.batch_size, self.hidden_size))
        self.yts = np.empty((self.batch_size, self.output_size))

        if self.memorize == False:
            self.previous_ht = np.zeros(self.hidden_size)
            
        for t in range(self.batch_size):
            xt  = input_tensor[t,]
            xt_tilda = np.concatenate((self.previous_ht.flatten(), xt.flatten())).reshape(1, -1)
            xt_tilda = self.fc_xt.forward(xt_tilda)

            xt_tilda_activation = self.fc_xt.input_tensor.copy()
            self.state[XT_TILDA_ACTIVATION][t] = xt_tilda_activation

            activated_xt_tilda = self.tanh_layer.forward(xt_tilda)

            tanh_activation = self.tanh_layer.activation.copy()  
            self.state[TANH_ACTIVATION][t] = tanh_activation

            self.previous_ht = activated_xt_tilda.copy()

            yt = self.fc_yt.forward(activated_xt_tilda)
            
            yt_activation = self.fc_yt.input_tensor.copy()
            self.state[YT_ACTIVATION][t] = yt_activation

            activated_yt = self.sigmoid_layer.forward(yt)

            sigmoid_activation = self.sigmoid_layer.activation.copy()
            self.state[SIGMOID_ACTIVATION][t] = sigmoid_activation

            self.hts[t] = self.previous_ht
            self.yts[t] = activated_yt

        return self.yts

    def backward(self, error_tensor):
        
        out_grad = np.zeros(self.input_tensor.shape)

        xt_grad  = np.zeros((self.fc_xt.weights.shape))
        yt_grad  = np.zeros((self.fc_yt.weights.shape))
        whh_grad = np.zeros((self.fc_xt.weights.shape))
        wxh_grad = np.zeros((self.fc_xt.weights.shape))

        next_ht  = 0 

        for t in reversed(range(self.batch_size)):
            yt_error = error_tensor[t,]

            self.sigmoid_layer.activation = self.state[SIGMOID_ACTIVATION][t]
            d_yt = self.sigmoid_layer.backward(yt_error)
            
            self.fc_yt.input_tensor = self.state[YT_ACTIVATION][t]
            d_yt = self.fc_yt.backward(d_yt.reshape(1, -1)) 
            
            yt_grad += self.fc_yt.gradient_weights
            delta_ht = d_yt + next_ht # Gradient of a copy procedure is a sum

            self.tanh_layer.activation = self.state[TANH_ACTIVATION][t]
            tanh_grad = self.tanh_layer.backward(delta_ht) 

            self.fc_xt.input_tensor = self.state[XT_TILDA_ACTIVATION][t] 
            error_grad = self.fc_xt.backward(tanh_grad)

            xt_tilda_activation = self.state[XT_TILDA_ACTIVATION][t].copy()
            whh_grad += np.dot(xt_tilda_activation.T, tanh_grad)

            next_ht     = error_grad[:, 0:self.hidden_size]
            out_grad[t] = error_grad[:, self.hidden_size:(self.input_size + self.hidden_size + 1)] 
            
            xt_grad  += self.fc_xt.gradient_weights

            wh = self.fc_xt.gradient_weights.copy()
            wxh_grad += np.dot(wh, tanh_grad.T)

        self.gradient_weights = whh_grad

        if self._optimizer is not None:
            self.fc_xt.weights = self._optimizer.calculate_update(self.fc_xt.weights, xt_grad)
            self.fc_yt.weights = self._optimizer_yt.calculate_update(self.fc_yt.weights, yt_grad)

        self.weights = self.fc_xt.weights
        
        return out_grad

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer     = deepcopy(optimizer)
        self._optimizer_yt  = deepcopy(optimizer)
        self.fc_xt.optimizer = None
        self.fc_yt.optimizer = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_yt.initialize(weights_initializer, bias_initializer)
        self.fc_xt.initialize(weights_initializer, bias_initializer)

    @property
    def weights(self):
        return self.fc_xt.weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def set_weight(self, weights):
        self.fc_xt.weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights