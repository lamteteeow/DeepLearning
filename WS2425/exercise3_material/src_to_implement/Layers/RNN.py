import numpy as np
import random
from Optimization.Optimizers import Sgd
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = None # h(t)
        self.hidden_state_last_sequence = np.zeros(self.hidden_size) # h(t-1)
        self.all_hidden_states = None # store all hidden states of the current sequence
        # Layer with trainalbe parameters
        self.trainable = True 
        # Property representing whether the RNN regards subsequent sequences as a belonging to the same long sequence.
        self.memorize = False
        # These FC layers used to compute hidden state and output
        # check the slides
        self.output_fc = FullyConnected(self.hidden_size, self.output_size)
        self.hidden_state_fc = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        # store activations of the FC layers for backwards
        self.output_fc_outputs = None
        self.hidden_state_fc_outputs = None
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.output_fc.initialize(weights_initializer, bias_initializer)
        self.hidden_state_fc.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        if self.optimizer is not None:
            if self.optimizer.regularizer is not None:
                return self.optimizer.regularizer.norm(self.weights)
        return 0.0
        
    def forward(self, input_tensor):
        time = input_tensor.shape[0] # treat batch as time
        self.input_tensor = input_tensor # Save value for backward
        self.output_tensor = np.zeros((time, self.output_size))
        self.all_hidden_states = np.zeros((time, self.hidden_size))
        self.output_fc_outputs = np.zeros((time, self.output_size))
        self.hidden_state_fc_outputs = np.zeros((time, self.hidden_size))
        if self.memorize == False:
            self.hidden_state = np.zeros((self.hidden_size,))
        else: 
            # restore hidden state from the last iteration: h(t-1)
            self.hidden_state = self.hidden_state_last_sequence
        self.first_hidden_state = self.hidden_state # for backward
        for t in range(time):
            xt = input_tensor[t]
            # at this point, self.hidden_state is h(t-1)
            x_tilde_t = np.hstack([xt, self.hidden_state])
            logits = self.hidden_state_fc.forward(x_tilde_t.reshape((1, -1)))
            self.hidden_state_fc_outputs[t] = logits.reshape((-1,))
            logits = self.tanh.forward(logits)
            # now self.hidden_state is h(t)
            self.hidden_state = logits.reshape((-1,))
            self.all_hidden_states[t] = self.hidden_state
            # now compute output
            out_logits = self.output_fc.forward(logits)
            self.output_fc_outputs[t] = out_logits.reshape((-1,))
            out_logits = self.sigmoid.forward(out_logits)
            self.output_tensor[t] = out_logits.reshape((-1,))
        # Save hidden state for next iteration in case we need it
        self.hidden_state_last_sequence = self.hidden_state

        return self.output_tensor
    
    def backward(self, error_tensor):
        time = error_tensor.shape[0]
        prev_error_tensor = np.zeros_like(self.input_tensor)
        self.gradient_weights = np.zeros_like(self.weights)
        self.output_fc_gradient_weights = np.zeros_like(self.output_fc.weights)
        # Backpropagation through time
        for t in reversed(range(time)):
            # Restore the state of the layers
            xt = self.input_tensor[t]
            cur_error_tensor = error_tensor[t]
            if t > 0:
                hprev = self.all_hidden_states[t-1]
            else:
                hprev = self.first_hidden_state
            self.hidden_state_fc.input_tensor = np.hstack([xt, hprev, np.ones((1,))]).reshape((1, -1))
            self.tanh.input_tensor = self.hidden_state_fc_outputs[t].reshape((1, -1))
            self.sigmoid.input_tensor = self.output_fc_outputs[t].reshape((1, -1))
            self.output_fc.input_tensor = np.hstack([np.tanh(self.hidden_state_fc_outputs[t]), np.ones((1,))]).reshape((1, -1))
            # Now begin BPTT
            output_fc_error_tensor = self.sigmoid.backward(cur_error_tensor)
            tanh_error_tensor = self.output_fc.backward(output_fc_error_tensor) # error tensor y_t
            # grad h_t = error_tensor ht + error_tensor yt, check exercise slides
            grad_h_t = tanh_error_tensor
            if t < time-1:
                grad_h_t += self.error_tensor_ht
            hidden_state_fc_error_tensor = self.tanh.backward(grad_h_t)
            xt_hprev_grad = self.hidden_state_fc.backward(hidden_state_fc_error_tensor)
            # Accumulate gradients for the two FC layers
            self.output_fc_gradient_weights += self.output_fc.gradient_weights
            self.gradient_weights += self.hidden_state_fc.gradient_weights
            # Return the error tensor to lower layers
            prev_error_tensor[t] = xt_hprev_grad[0][:self.input_size]
            self.error_tensor_ht = xt_hprev_grad[0][self.input_size:]
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.output_fc.weights = self.optimizer.calculate_update(self.output_fc.weights, self.output_fc_gradient_weights)
        return prev_error_tensor

    @property
    def optimizer(self): #getter
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, new_value): #setter new value
        self._optimizer = new_value

    @property
    def weights(self): #getter
        return self.hidden_state_fc.weights
    
    @weights.setter
    def weights(self, new_value):
        self.hidden_state_fc.weights = new_value

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
        