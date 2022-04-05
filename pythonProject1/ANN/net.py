import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.softmax(out)
        out = self.predict(out)

        return out

# 定义网络参数
EPOCH = 50
BATCH_SIZE = 20
LR = 0.03


net = Net(6, 32, 3)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

# 导入数据，数据可以是多种格式，输入路径后利用pandas读取数据
features = pd.read_excel('ur_url')
# 标签记录了不同产状的滑石类型：0代表斑状；1代表透镜状；2代表层状
labels = np.array(features['type'])
# 利用.drop去掉多余列
features = features.drop('type', axis=1)
features = np.array(features)
# Numpy需要转化为torch中的tensor
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25)
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
train_x = train_x.to(torch.float32)
test_x = test_x.to(torch.float32)

for ep in range(EPOCH):
    for data in range(len(train_x)):  # 对于训练集的每一个batch
        out = net(train_x)  # 送进网络进行输出
        loss = loss_function(out, train_y)  # 获得损失

        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播获得梯度
        optimizer.step()  # 更新梯度

    num_correct = 0  # 正确分类的个数，在测试集中测试准确率
    total = 0
    for data in range(len(test_x)):
        out = net(test_x)  # 送进网络进行输出

        out1 = net(test_x)  # 获得输出

        _, prediction = torch.max(out1, 1)
    num_correct += (prediction == test_y).sum().item() # 找出预测和真实值相同的数量，也就是以预测正确的数量
    total += test_y.size(0)
    accuracy = num_correct / total # 计算正确率
    timeSpan = time.perf_counter()
    print("epoch:%d，acc:%f,time: %dS" % (ep + 1, accuracy, timeSpan))