from NeuralNetwork import NeuralNetwork
from Layers import *
from Optimization import *


def build():
    optimizer = Optimizers.Adam(learning_rate=5e-4, mu=0.9, rho=0.999)
    optimizer.add_regularizer(Constraints.L2_Regularizer(alpha=4e-4))

    LeNet = NeuralNetwork(
        optimizer=optimizer,
        weights_initializer=Initializers.He(),
        bias_initializer=Initializers.He(),
    )

    (
        LeNet
        >> Conv.Conv(stride_shape=(1, 1), convolution_shape=(1, 5, 5), num_kernels=6)
        >> ReLU.ReLU()
        >> Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2))
        >> Conv.Conv(stride_shape=(1, 1), convolution_shape=(1, 5, 5), num_kernels=16)
        >> ReLU.ReLU()
        >> Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2))
        >> Flatten.Flatten()
        >> FullyConnected.FullyConnected(input_size=16 * 7 * 7, output_size=120)
        >> ReLU.ReLU()
        >> FullyConnected.FullyConnected(input_size=120, output_size=84)
        >> ReLU.ReLU()
        >> FullyConnected.FullyConnected(input_size=84, output_size=10)
        >> SoftMax.SoftMax()
    )

    LeNet.loss_layer = Loss.CrossEntropyLoss()

    return LeNet


# LetNet-5 Architecture: (On MNIST)
#   >> [batch_size x 1  x 28 x 28] | MNIST Input

#   >> [batch_size x 6  x 28 x 28] | Convolution with stride = 1, kernel = [1 x 5 x 5], number of kernel = 6
#   >> [batch_size x 6  x 28 x 28] | ReLU Activation
#   >> [batch_size x 6  x 14 x 14] | Max-Pooling with stride = 2, kernel = [2 x 2]

#   >> [batch_size x 16 x 14 x 14] | Convolution with stride = 1, kernel = [1 x 5 x 5], number of kernel = 16
#   >> [batch_size x 16 x 14 x 14] | ReLU Activation
#   >> [batch_size x 16 x 7  x 7 ] | Max-Pooling with stride = 2, kernel = [2 x 2]

#   >> [batch_size x 784]          | Flatten 16 x 7 x 7 = 784

#   >> [batch_size x 120]          | FC Layer
#   >> [batch_size x 120]          | ReLU Activation

#   >> [batch_size x 84 ]          | FC Layer
#   >> [batch_size x 84 ]          | ReLU Activation

#   >> [batch_size x 10 ]          | FC Layer
#   >> [batch_size x 10 ]          | Softmax Layer
