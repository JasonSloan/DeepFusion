# 非结构细粒度剪枝
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def prune_conv_layer(layer, prune_rate):
    # 取出卷积层的权重
    weight = layer.weight.data.cpu().numpy()    
    print(weight.shape) # (64, 3, 3, 3) (卷积核的数量，卷积核的深度，卷积核的大小，卷积核的大小)
    # 获得卷积权重的总数量
    num_weights = weight.size
    # 获得需要剪枝的数量
    num_prune = int(num_weights * prune_rate)
    # 将权重展开成一维
    flat_weights = np.abs(weight.flatten())
    # 计算按照prune_rate需要剪枝的阈值
    threshold = np.sort(flat_weights)[num_prune]
    # 将小于阈值的权重置为0(这里还有不合理的地方,因为排序时使用的是绝对值,但是重新赋值时没有绝对值)
    weight[weight < threshold] = 0
    # 将剪枝后的权重重新赋值给卷积层
    layer.weight.data = torch.from_numpy(weight).to(layer.weight.device)



if __name__ == '__main__':
    net = Net()
    prune_rate = 0.2
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            prune_conv_layer(module, prune_rate)


