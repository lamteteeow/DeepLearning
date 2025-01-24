import numpy as np
from Layers.Base import BaseLayer

# Flatten layers reshapes the multi-dimensional input to a one dimensional feature vector.
class Flatten(BaseLayer): # convert data into a 1-D array
    def __init__(self):
        super().__init__() #super constructor

    def forward(self, input_tensor):
        # input_tensor brings batch_size as rows, 
        # batch size x input size
        self.batch_size = input_tensor.shape[0]
        # to take elements starting from index 1 up to (but not including) index 4
        self.input_shape = input_tensor.shape[1:10]  # take all the size of input, except the batch size
        # reshapes and returns the input tensor. refer line 483
        # output_tensor = flatten.forward(self.input_tensor)
        # input_vector = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        # prod: the product of array elements over a given axis
        # each batch, flatten with the length by the value of prod(input_shape)
        output =input_tensor.reshape(self.batch_size, np.prod(self.input_shape))

        #self.input_shape = input_tensor.shape
        #return np.reshape(input_tensor, (self.input_shape[0], -1))
        return output # input tensor for Conv

    def backward(self, error_tensor): # error_tensor = batch_size x output_size
        # weights = input x output => error_previous = batch_sizex(out_sizexout_size)xinput_size
        #error tensor for previous layer has a shape of (batch x input)
        # desired output shape is batches x channels x spatialX x (spatialY) -refer line 639
        # * to take all elements of an array input shape, just number
        # bring back those value of input_shape before producing, from forward part
        output = error_tensor.reshape(self.batch_size, *self.input_shape)
        return output # error tensor
        # return np.reshape(error_tensor, self.input_shape)
