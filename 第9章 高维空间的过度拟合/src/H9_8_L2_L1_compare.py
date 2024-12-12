import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

if __name__=="__main__":

    name = [
        "L1_model_Linear_0.txt", 
        "L1_model_Linear_1.txt", 
        "L1_model_Linear_2.txt",
        "L1_model_Linear_3.txt"
    ]
    # name = [
    #     "L2_model_Linear_0.txt", 
    #     "L2_model_Linear_1.txt", 
    #     "L2_model_Linear_2.txt",
    #     "L2_model_Linear_3.txt"
    # ]

    # name = [
    #     "over_fitting_model_Linear_0.txt", 
    #     "over_fitting_model_Linear_1.txt", 
    #     "over_fitting_model_Linear_2.txt",
    #     "over_fitting_model_Linear_3.txt"
    # ]

    total = 0
    total_small = 0
    total_tiny = 0
    for i in range(4):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), "model", name[i])
        WB = np.loadtxt(file_path)
        W = WB[0:-1]
        
        total += np.prod(W.shape)

        norm_1 = np.linalg.norm(W, 1)
        max_norm1 = np.max(norm_1)
        min_norm1 = np.min(norm_1)

        norm_2 = np.linalg.norm(W, 2)
        max_norm2 = np.max(norm_2)
        min_norm2 = np.min(norm_2)

        num_small = np.sum(np.abs(W) < 1e-1)
        total_small += num_small
        num_tiny = np.sum(np.abs(W) < 1e-5)
        total_tiny += num_tiny

        print("layer,norm1,norm2", i, norm_1, norm_2)
    print("total, small, tiny",total, total_small, total_tiny)
