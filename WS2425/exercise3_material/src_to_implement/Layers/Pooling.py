from Layers.Base import BaseLayer, PhaseSeperatableLayer
import numpy as np

class Pooling(PhaseSeperatableLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape= stride_shape
        self.pooling_shape= pooling_shape

    def forward(self, input_tensor):
        self.input_tensor_shape  = input_tensor.shape
        b_size, C_in, H_in, W_in = input_tensor.shape
        P_h, P_w = self.pooling_shape
        S_h, S_w = self.stride_shape
        
        H_out = np.floor((H_in - P_h) / S_h).astype(int) + 1
        W_out = np.floor((W_in - P_w) / S_w).astype(int) + 1

        output = np.zeros((b_size, C_in, H_out, W_out))

        self.pooling_locations = list()

        for b_idx in range(b_size):
            for c_idx in range(C_in):
                for pool_window, (h_idx, w_idx) in self.slide_window(
                    input_tensor[b_idx, c_idx], 
                    H_out, 
                    W_out, 
                    *self.stride_shape, 
                    *self.pooling_shape
                ):
                    max_value = np.max(pool_window)
                    (mxi, mxj), *_ = np.argwhere(pool_window == max_value) + [h_idx * S_h, w_idx * S_w]
                    
                    output[b_idx, c_idx, h_idx, w_idx] = max_value                    
                    self.pooling_locations.append((b_idx, c_idx, mxi, mxj)) 

        return output

    def backward(self, error_tensor):
        *_, H_in, W_in = error_tensor.shape

        output = np.zeros((self.input_tensor_shape)).astype(float)
        b_size, C_in, *_ = output.shape
        
        pos_idx = 0
        for b_idx in range(b_size):
            for c_idx in range(C_in):
                for h_idx in range(H_in):
                    for w_idx in range(W_in): 
                        *_, mxi, mxj = self.pooling_locations[pos_idx]
                        output[b_idx, c_idx, mxi, mxj] += error_tensor[b_idx, c_idx, h_idx, w_idx]
                        pos_idx += 1
                        
        return output
    
    def slide_window(self, input_tensor, H_out, W_out, S_h, S_w, P_h, P_w):
        for i in range(0, H_out):
            for j in range(0, W_out):
                tensor_window = input_tensor[i * S_h: i * S_h + P_h, 
                                             j * S_w: j * S_w + P_w]
                yield tensor_window, (i, j)
        




