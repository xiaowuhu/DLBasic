
import numpy as np
import common.Layers as layer
import common.ConvLayer as conv_layer

if __name__=="__main__":
    conv = layer.Conv1d((2,5), (3,4), stride=1)
 #   x = np.array([[1.0,2,3,4.0,5],[1,3,4,2,2],[1,1,2,1,0]]).reshape(3,1,1,5)
    x = np.array([[1.0,2,3,4.0,5],[1,3,4,2,2],[1.0,2,3,4.0,5],[1,3,4,2,2]]).reshape(2,2,5)
    result = conv.forward(x)
    print("forward:\n", result)
    print(result.shape)
    #delta_in = np.array([[1,2],[3,2],[3,1],[1,1],[0,0],[2,1]]).astype(np.float64).reshape(3,2,1,2)
    delta_in = np.ones(result.shape)
    result = conv.backward(delta_in)
   #  print("back:\n", result)
   #  print(result.shape)
    print("------------")
    conv1 = conv_layer.ConvLayer((2, 1, 5), (3, 1, 4), (1, 0), ("normal","SDG", 0.1))
    conv1.initialize()
    conv1.set_filter(conv.WB.W, conv.WB.B)
    result = conv1.forward(x)
    print("forward:\n", result)
    print(result.shape)
    result = conv1.backward(delta_in, 0)
    print("back:\n", result)
    print(result.shape)
