
# 模型基类
class Module(object):
    def __init__(self):
        pass
    
    def forward(self, X):
        pass

    def backward(self, X, Y, Z):
        pass

    def predict(self, X):
        return self.forward(X)

    # 目前这个operators必须有,否则从locals()中获取不到
    def save_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id) + ".txt"
            op.save(op_name)
            unique_id += 1

    def load_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id) + ".txt"
            op.load(op_name)
            unique_id += 1
