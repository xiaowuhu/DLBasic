# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import math
from pathlib import Path
import numpy as np

class WeightsBias_2_1(object):
    def __init__(self, n_input, n_output, init_method, optimizer_name, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.learning_rate = eta

    def Initialize(self, folder, name, create_new):
        self.init_file_name = str.format("{0}/{1}_{2}_{3}_init.npz", folder, name, self.num_input, self.num_output)
        self.result_file_name = str.format("{0}/{1}_result.npz", folder, name)
        
        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameters()
        
        # end if
        self.CreateOptimizers()

        self.dW = np.zeros(self.W.shape).astype('float32')
        self.dB = np.zeros(self.B.shape).astype('float32')

    def CreateNew(self):
        self.W, self.B = WeightsBias_2_1.InitialParameters(self.num_input, self.num_output, self.init_method)
        #self.SaveInitialValue()
        
    def LoadExistingParameters(self):
        w_file = Path(self.init_file_name)
        if w_file.exists():
            self.LoadInitialValue()
        else:
            self.CreateNew()
        # end if
    
    
    def Update(self):
        self.W = self.oW.update(self.W, self.dW)
        self.B = self.oB.update(self.B, self.dB)

    def SaveInitialValue(self):
        np.savez(self.init_file_name, W=self.W, B=self.B)

    def LoadInitialValue(self):
        data = np.load(self.init_file_name)
        self.W = data["W"]
        self.B = data["B"]

    def SaveResultValue(self):
        np.savez(self.result_file_name, W=self.W, B=self.B)

    def LoadResultValue(self):
        data = np.load(self.result_file_name)
        self.W = data["W"]
        self.B = data["B"]

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == "zero":
            W = np.zeros((num_input, num_output)).astype('float32')
        elif method == "normal":
            W = np.random.normal(size=(num_input, num_output)).astype('float32')
        elif method == "kaiming":
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output)).astype('float32')
        elif method == "xavier":
            t = math.sqrt(6/(num_output+num_input))
            W = np.random.uniform(-t, t, (num_input, num_output)).astype('float32')
        # end if
        B = np.zeros((1, num_output)).astype('float32')
        return W, B
