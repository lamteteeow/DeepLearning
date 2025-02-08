import numpy as np
from Layers import FullyConnected
from Layers import Sigmoid
from Layers import TanH
from Layers.Base import BaseLayer

from copy import deepcopy


class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = None
        self.memorize = False
        self.input_tensor = None
        self.cell_state = None

        # fullyconnected for forget, input, c~, o gate
        self.FC_fico = FullyConnected.FullyConnected(
            self.input_size + self.hidden_size, self.hidden_size * 4
        )
        # fullyconnected for output
        self.FC_y_t = FullyConnected.FullyConnected(hidden_size, output_size)

        self.sigmoid = Sigmoid.Sigmoid()
        self.tanh = TanH.TanH()
        self.optimizer = None
        self.gradient_weights = None
        # self.weights = None

    def initialize(self, weights_init, bias_init):
        self.FC_fico.initialize(weights_init, bias_init)
        self.FC_y_t.initialize(weights_init, bias_init)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]

        self.forget_gate = np.zeros((self.batch_size, self.hidden_size))
        self.input_gate = np.zeros_like(self.forget_gate)
        self.cell_tilde = np.zeros_like(self.forget_gate)
        self.o_gate = np.zeros_like(self.forget_gate)

        self.input_hidden_tensor = np.zeros((self.batch_size, self.hidden_size + self.input_size))
        self.y_t = np.zeros((self.batch_size, self.output_size))

        self.input_tensor_FC_y = []
        self.input_tensor_FC_fico = []

        if self.hidden_state is None:
            self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
        if self.cell_state is None:
            self.cell_state = np.zeros((self.batch_size + 1, self.hidden_size))

        if self.memorize is False:
            self.hidden_state[self.batch_size, :] = np.zeros((1, self.hidden_size))
            self.cell_state[self.batch_size, :] = np.zeros((1, self.hidden_size))
        else:
            self.hidden_state[self.batch_size, :] = self.hidden_state[self.batch_size - 1, :]
            self.cell_state[self.batch_size, :] = self.cell_state[self.batch_size - 1, :]

        for t in range(self.batch_size):
            self.input_hidden_tensor[t, :] = np.concatenate(
                (self.hidden_state[t - 1, :], self.input_tensor[t, :])
            )
            temporal_array = np.array([self.input_hidden_tensor[t, :]])
            output_FC_fico = self.FC_fico.forward(temporal_array)
            self.input_tensor_FC_fico.append(self.FC_fico.input_tensor_w_bias)
            output_sigmoid = self.sigmoid.forward(output_FC_fico)

            self.forget_gate[t, :] = np.array([output_sigmoid[:, 0 : self.hidden_size]])
            self.input_gate[t, :] = np.array(
                output_sigmoid[:, self.hidden_size : 2 * self.hidden_size]
            )
            self.o_gate[t, :] = np.array(
                [output_sigmoid[:, 2 * self.hidden_size : 3 * self.hidden_size]]
            )
            self.cell_tilde[t, :] = self.tanh.forward(
                output_FC_fico[:, 3 * self.hidden_size : 4 * self.hidden_size]
            )

            self.cell_state[t, :] = (self.forget_gate[t, :] * self.cell_state[t - 1, :]) + (
                self.input_gate[t, :] * self.cell_tilde[t, :]
            )
            self.hidden_state[t, :] = self.tanh.forward(self.cell_state[t, :]) * self.o_gate[t, :]

            temporal_array = np.array([self.hidden_state[t, :]])
            self.y_t[t, :] = self.sigmoid.forward(self.FC_y_t.forward(temporal_array))
            self.input_tensor_FC_y.append(self.FC_y_t.input_tensor_w_bias)

        return self.y_t

    def backward(self, error_tensor):
        dh_previous = np.array([np.zeros_like(self.hidden_state[self.batch_size, :])])
        dc_previous = np.array([np.zeros_like(self.cell_state[self.batch_size, :])])

        d_input_tensor = np.zeros_like(self.input_hidden_tensor)

        self.dW_y = np.zeros_like(self.FC_y_t.weights)
        self.dw_o = np.zeros_like(self.FC_fico.weights[:, 0 : self.hidden_size])
        self.dw_c_tilde = np.zeros_like(self.FC_fico.weights[:, 0 : self.hidden_size])
        self.dw_i = np.zeros_like(self.FC_fico.weights[:, 0 : self.hidden_size])
        self.dw_f = np.zeros_like(self.FC_fico.weights[:, 0 : self.hidden_size])

        self.dw_x = np.zeros_like(self.FC_fico.weights)

        temporal_weight = self.FC_fico.weights

        for t in range(self.batch_size)[::-1]:
            # gradient for output
            self.sigmoid.fx = self.y_t[t, :]
            # self.FC_y_t.input_tensor = np.array([self.hidden_state[t,:]])
            # Z = self.sigmoid.backward(error_tensor[t,:]) * error_tensor[t,:]

            Z = self.sigmoid.backward(error_tensor[t, :])
            self.FC_y_t.input_tensor_w_bias = self.input_tensor_FC_y[t]
            dh_t = self.FC_y_t.backward(np.array([Z])) + dh_previous

            self.dW_y += self.FC_y_t.gradient_weights

            do_t = dh_t * self.tanh.forward(self.cell_state[t, :])
            dc_t = self.tanh.backward(error_tensor=dh_t * self.o_gate[t, :]) + dc_previous
            # dc_t = self.tanh.backward(dh_t) * self.o_gate[t, :] + dc_next

            dc_previous = self.forget_gate[t, :] * dc_t
            df_t = self.cell_state[t - 1, :] * dc_t
            di_t = self.cell_tilde[t, :] * dc_t
            dc_tilde_t = self.input_gate[t, :] * dc_t

            # gate o
            self.sigmoid.fx = self.o_gate[t, :]
            error_o_gate = self.sigmoid.backward(do_t)

            self.FC_fico.weights = temporal_weight[:, 2 * self.hidden_size : 3 * self.hidden_size]
            self.FC_fico.input_tensor_w_bias = self.input_tensor_FC_fico[t]
            dX_o = self.FC_fico.backward(error_o_gate)
            self.dw_o += self.FC_fico.gradient_weights

            # gate c_tilde
            self.tanh.fx = self.cell_tilde[t, :]
            error_c_tilde = self.tanh.backward(dc_tilde_t)

            self.FC_fico.weights = temporal_weight[:, 3 * self.hidden_size : 4 * self.hidden_size]
            self.FC_fico.input_tensor_w_bias = self.input_tensor_FC_fico[t]
            dX_c_tilde = self.FC_fico.backward(error_c_tilde)
            self.dw_c_tilde += self.FC_fico.gradient_weights

            # gate input
            self.sigmoid.fx = self.input_gate[t, :]
            error_i = self.sigmoid.backward(di_t)

            self.FC_fico.weights = temporal_weight[:, self.hidden_size : 2 * self.hidden_size]
            self.FC_fico.input_tensor_w_bias = self.input_tensor_FC_fico[t]
            dX_i = self.FC_fico.backward(error_i)
            self.dw_i += self.FC_fico.gradient_weights

            # gate forget
            self.sigmoid.fx = self.forget_gate[t, :]
            error_f = self.sigmoid.backward(df_t)

            self.FC_fico.weights = temporal_weight[:, 0 : self.hidden_size]
            self.FC_fico.input_tensor_w_bias = self.input_tensor_FC_fico[t]
            dX_f = self.FC_fico.backward(error_f)
            self.dw_f += self.FC_fico.gradient_weights

            d_input_tensor[t, :] = dX_f + dX_i + dX_o + dX_c_tilde
            # self.FC_fico.input_tensor_w_bias = self.input_tensor_FC_fico[t]
            # self.FC_fico.weights = temporal_weight
            # d_input_tensor[t,:] = self.FC_fico.backward(np.concatenate((error_f, error_i, error_c_tilde, error_o_gate), axis= 1))
            dh_previous = d_input_tensor[t, 0 : self.hidden_size]

        self.FC_fico.weights = temporal_weight
        self.dw_x = np.concatenate((self.dw_f, self.dw_i, self.dw_o, self.dw_c_tilde), axis=1)
        self.gradient_weights = self.dw_x

        if self.optimizer is not None:
            self.FC_fico.weights = self.optimizer.calculate_update(
                self.weights, self.gradient_weights
            )
            self.FC_y_t.weights = self.optimizer.calculate_update(self.FC_y_t.weights, self.dW_y)

        return d_input_tensor[:, self.hidden_size : self.hidden_size + self.input_size]

    def set_optimizer(self, x):
        self.optimizer = x

    def get_optimizer(self):
        return self.optimizer

    @property
    def weights(self):
        return deepcopy(self.FC_fico.weights)

    @weights.setter
    def weights(self, weight_tensor):
        self.FC_fico.weights = weight_tensor
