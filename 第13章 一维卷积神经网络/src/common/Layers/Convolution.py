import numpy as np
from .Operator import Operator
from .WeightsBias import WeightsBias


# 一维卷积
class Conv1d(Operator):
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride=1, padding=0,
                 init_method: str="normal", 
                 optimizer: str="SGD"):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.WB = WeightsBias(kernel_size, 1, init_method, optimizer)
        self.WB.W = self.WB.W.T
        self.WB.dW = self.WB.dW.T
        self.stride = stride
        self.padding = padding
        self.output_size = None
        self.input_size = None

    def get_parameters(self):
        return self.WB

    def _conv1d(self, data, kernal, output_size, kernal_size, stride=1):
        assert(kernal.shape[1] == kernal_size)
        result = np.zeros((1, output_size))
        for i in range(output_size):
            ii = i * stride
            data_window = data[:, ii:ii+kernal_size]
            result[0, i] = np.dot(data_window, kernal.T) # np.sum(np.multily(a,b))
        return result

    def forward(self, batch_x):
        self.batch_input = batch_x
        self.m = batch_x.shape[0]
        if self.output_size is None:
            self.output_size = 1 + (self.batch_input.shape[1] + 2 * self.padding - self.kernel_size)//self.stride
        if self.input_size is None:
            self.input_size = self.batch_input.shape[1]
        self.batch_output = np.zeros((self.batch_input.shape[0], self.output_size))
        for i in range(self.m):
            data = self.batch_input[i:i+1]
            self.batch_output[i] = self._conv1d(data, self.WB.W, self.output_size, self.kernel_size, self.stride)
            
        return self.batch_output

    def backward(self, delta_in):
        if self.stride > 1:
            delta_in_stride = self.expand_delta_map(
                delta_in, self.m, 
                self.in_channel, 1, 
                self.input_size, 1, self.output_size,
                1, self.kernel_size,
                self.padding, self.stride)
        else:
            delta_in_stride = delta_in

        # 求权重梯度
        self.WB.dW *= 0
        for i in range(self.m):
            data = self.batch_input[i:i+1]
            filter = delta_in_stride[i:i+1]
            self.WB.dW += self._conv1d(data, filter, self.kernel_size, delta_in_stride.shape[1])
        self.WB.dW /= self.m
        #self.WB.dB = np.sum(delta_in, keepdims=True) / self.m
        #print(self.WB.dW, self.WB.dB)


        # 求输出梯度
        pad_left,pad_right = self.calculate_padding_size(
            delta_in_stride.shape[1], self.kernel_size, self.input_size)
        delta_in_pad = np.pad(delta_in_stride,((0,0),(pad_left,pad_right)))
        delta_out = np.zeros(self.batch_input.shape)
        filter = np.flip(self.WB.W)
        for i in range(self.m):
            data = delta_in_pad[i:i+1]
            delta_out[i] = self._conv1d(data, filter, self.input_size, self.kernel_size)
        return delta_out

    def load(self, name):
        WB = super().load_from_txt_file(name)
        if WB.ndim == 1:
            WB = np.expand_dims(WB, axis=1)
        self.WB.set_WB(WB)
        self.WB.W = self.WB.W.T
    
    def save(self, name):
        W, B = self.WB.get_WB()
        WB = np.concatenate((W.T, B))
        super().save_to_txt_file(name, WB)

    def calculate_padding_size(self, input_size, kernal_size, output_size, stride=1):
        pad_w = ((output_size - 1) * stride - input_size + kernal_size) // 2
        return (pad_w, pad_w)

    # stride 不为1时，要先对传入的误差矩阵补十字0
    def expand_delta_map(self, dZ, batch_size, input_c, input_h, input_w, output_h, output_w, filter_h, filter_w, padding, stride):
        expand_h = 0
        expand_w = 0
        if stride == 1:
            dZ_stride_1 = dZ
            expand_h = dZ.shape[2]
            expand_w = dZ.shape[3]
        else:
            # 假设如果stride等于1时，卷积后输出的图片大小应该是多少，然后根据这个尺寸调整delta_z的大小
            (expand_h, expand_w) = self.calculate_output_size(input_h, input_w, filter_h, filter_w, padding, 1)
            # 初始化一个0数组，四维
            #dZ_stride_1 = np.zeros((batch_size, input_c, expand_h, expand_w)).astype(np.float32)
            dZ_stride_1 = np.zeros((batch_size, expand_w))
            # 把误差值填到当stride=1时的对应位置上
            for bs in range(batch_size):
                #for ic in range(input_c):
                    #for i in range(output_h):
                for j in range(output_w):
                    #ii = i * stride
                    jj = j * stride
                    #dZ_stride_1[bs, ic, ii, jj] = dZ[bs, ic, i, j]
                    dZ_stride_1[bs, jj] = dZ[bs, j]
                        #end j
                    # end i
                # end ic
            # end bs
        # end else
        return dZ_stride_1

    def calculate_output_size(self, input_h, input_w, filter_h, filter_w, padding, stride=1):
        output_h = (input_h - filter_h + 2 * padding) // stride + 1    
        output_w = (input_w - filter_w + 2 * padding) // stride + 1
        return (output_h, output_w)
