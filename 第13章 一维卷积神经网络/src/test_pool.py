
import numpy as np
import common.Layers as layer

if __name__=="__main__":
    input_channel = 2
    batch = 1
    stride = 2
    pool_length = 2
    pool = layer.Pool1d((input_channel, 6), pool_length, stride, 0)
    x = np.array([0.1,1,3,2,4,5,2,3,4,2,1,3]).astype(np.float64).reshape(batch,input_channel,6)
    z = pool.forward(x)
    print(z)


    delta = np.array([1,2,3,4,5,6]).astype(np.float64).reshape(batch,input_channel, pool.output_length)
    delta = pool.backward(delta)
    print(delta)
