import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

XT_TILDA_ACTIVATION = "XT_TILDA_ACTIVATION"
YT_ACTIVATION = "YT_ACTIVATION"
TANH_ACTIVATION = "TANH_ACTIVATION"
SIGMOID_ACTIVATION = "SIGMOID_ACTIVATION"


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.trainable = True
        self.memorize = False
        self.regularizer = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state_list = None

        self.fc_xt = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_yt = FullyConnected(self.hidden_size, self.output_size)

        self.optimizer = None
        self.fc_xt.optimizer, self.fc_yt.optimizer = None, None

        # self.weights = np.random.uniform(0.0, 1.0, (self.fc_xt.weights.shape))
        self.weights = self.fc_xt.weights

        self.hidden_state = np.zeros(hidden_size)  # h(t-1)

        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.state = {
            XT_TILDA_ACTIVATION: {},
            YT_ACTIVATION: {},
            TANH_ACTIVATION: {},
            SIGMOID_ACTIVATION: {},
        }

    def forward(self, input_tensor):
        time = input_tensor.shape[0]  # batch ~ time
        self.input_tensor = input_tensor.copy()
        self.output_tensor = np.zeros((time, self.output_size))
        self.hidden_state_list = np.zeros((time, self.hidden_size))

        if not self.memorize:
            self.hidden_state = np.zeros((self.hidden_size))
        self.first_hidden_state = self.hidden_state  # for backward

        for t in range(time):
            xt = input_tensor[t,]
            xt_tilda = np.concatenate((self.hidden_state.flatten(), xt.flatten()))
            logits = self.fc_xt.forward(xt_tilda.reshape(1, -1))

            logits = self.tanh.forward(logits)

            xt_tilda_activation = self.fc_xt.input_tensor.copy()
            self.state[XT_TILDA_ACTIVATION][t] = xt_tilda_activation

            activate_xt_tilda = self.tanh.forward(logits)
            self.state[TANH_ACTIVATION][t] = activate_xt_tilda.copy()

            self.hidden_state = activate_xt_tilda.copy()

            yt = self.fc_yt.forward(activate_xt_tilda)
            self.state[YT_ACTIVATION][t] = self.fc_yt.input_tensor.copy()

            activated_yt = self.sigmoid.forward(yt)
            self.state[SIGMOID_ACTIVATION][t] = activated_yt.copy()

            self.hidden_state_list[t] = self.hidden_state.copy()
            self.output_tensor[t] = activated_yt.copy()

        return self.output_tensor

    def backward(self, error_tensor):
        time = error_tensor.shape[0]  # batch ~ time
        out_grad = np.zeros(self.input_tensor.shape)

        xt_grad = np.zeros((self.fc_xt.weights.shape))
        yt_grad = np.zeros((self.fc_yt.weights.shape))
        whh_grad = np.zeros((self.fc_xt.weights.shape))
        wxh_grad = np.zeros((self.fc_xt.weights.shape))

        next_ht = 0

        for t in reversed(range(time)):
            yt_error = error_tensor[t,]

            self.sigmoid.activation = self.state[SIGMOID_ACTIVATION][t]
            d_yt = self.sigmoid.backward(yt_error)

            self.fc_yt.input_tensor = self.state[YT_ACTIVATION][t]
            d_yt = self.fc_yt.backward(d_yt.reshape(1, -1))

            yt_grad += self.fc_yt.gradient_weights
            delta_ht = d_yt + next_ht  # Gradient of a copy procedure is a sum

            self.tanh.activation = self.state[TANH_ACTIVATION][t]
            tanh_grad = self.tanh.backward(delta_ht)

            self.fc_xt.input_tensor = self.state[XT_TILDA_ACTIVATION][t]
            error_grad = self.fc_xt.backward(tanh_grad)

            whh_grad += np.dot(self.state[XT_TILDA_ACTIVATION][t].copy().T, tanh_grad)

            next_ht = error_grad[:, 0 : self.hidden_size]
            out_grad[t] = error_grad[:, self.hidden_size : (self.input_size + self.hidden_size + 1)]

            xt_grad += self.fc_xt.gradient_weights

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
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        self.__optimizer = val

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_xt.initialize(weights_initializer, bias_initializer)
        self.fc_yt.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self.__memorize

    @memorize.setter
    def memorize(self, val):
        self.__memorize = val

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self.__gradient_weights = val

    @property
    def weights(self):
        return self.fc_xt.weights

    @weights.setter
    def weights(self, val):
        self.fc_xt.weights = val