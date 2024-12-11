from H4_1_ShowData import load_data
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# def load_data():
#     file_name = "train4.txt"
#     file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
#     data_loader = DataLoader_4(file_path)
#     data_loader.load_data([0, 1])
#     #data_loader.normalize_train_data()  # 在本例中不需要归一化
#     data_loader.shuffle_data()
#     data_loader.split_data(0.8)
#     return data_loader

# def train_loop(data_loader: DataLoader_4, epoch=100):
#     batch_size = 1
#     lr = 0.5
#     W = np.zeros((1,1))
#     B = np.zeros((1,1))
#     nn = NeuralNet_4_7(data_loader, W, B, lr=lr, batch_size=batch_size)
#     check_data = nn.train(epoch, checkpoint=200)
#     return nn, check_data


# n1 - 正类样本数量，n2 - 负类样本数量
def show_result(X, Y, n1, n2, w, b, x):
    x1 = X[Y==1] 
    x2 = X[Y==0]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[0:n1], [0]*n1, c='r', marker='x', label='学区房')
    plt.scatter(x2[0:n2], [0]*n2, c='b', marker='o', label='普通房')
    plt.grid()
    plt.legend(loc='upper right')
    # 画出分界点
    if x is not None:
        plt.scatter(x, 0, marker="*")
    # 画出分界线
    if w is not None:
        minv = np.min(X)
        maxv = np.max(X)
        plt.plot([minv,maxv],[minv*w+b,maxv*w+b])

    plt.show()

# def train():
#     epoch = 100
#     # 准备数据
#     print("加载数据...")
#     data_loader = load_data()
#     print("训练神经网络...")
#     nn, check_data = train_loop(data_loader, epoch)
#     print("权重值 W =", nn.W)
#     print("偏置值 B =", nn.B)
#     weight = nn.W[0, 0]
#     bias = nn.B[0, 0]
#     x = - bias / weight
#     print("安居房和商品房的单价的分界点为", x)
#     check_data = np.array(check_data)
#     return check_data

W_POS = 0
B_POS = 1
DW_POS = 2
DB_POS = 3
X_POS = 4
Y_POS = 5

if __name__ == '__main__':    
    #check_data = train()
    #np.savetxt("check_data.txt", check_data, fmt="%.4f")
    check_data = load_data("check_data.txt")
    np.set_printoptions(precision=4)
    print(check_data)
    split_points = - check_data[:, B_POS] / check_data[:, W_POS]
    minv = np.min(check_data[:, 4]-0.1)
    maxv = np.max(check_data[:, 4]+0.1)
    print(split_points)

    markers = ['o', 'x']
    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.scatter(check_data[i, X_POS], 0, marker=markers[(int)(check_data[i, Y_POS])])  # 绘制样本点
        ax.text(check_data[i, X_POS]+0.05, 0+0.1, "$x_%i$"%(i+1))
        ax.scatter(split_points[i], 0, marker="s")  # 绘制分界点1
        #ax.text(split_points[i], 0.0, "$%i$"%(i+1))
        ax.plot([minv, maxv], # 绘制分割线
                [minv * check_data[i, W_POS] + check_data[i, B_POS], 
                maxv * check_data[i, W_POS] + check_data[i, B_POS],
                ], linestyle="solid")
        
        ax.scatter(split_points[i+1], 0, marker="s")  # 绘制分界点2
        #ax.text(split_points[i+1], 0, "$%i$"%(i+2))
        ax.plot([minv, maxv], # 绘制分割线
                [minv * check_data[i+1, W_POS] + check_data[i+1, B_POS], 
                maxv * check_data[i+1, W_POS] + check_data[i+1, B_POS],
                ], linestyle="dotted")
        #ax.grid()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("样本点 $x_%i$ 的分界线和分界点"%(i+1))

    plt.show()
