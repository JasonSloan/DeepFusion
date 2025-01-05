# 基于权重与梯度幅值之积的修建

import numpy as np
import torch


def prune_by_gradient_weight_product(model, pruning_rate):
    grad_weight_product_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 计算梯度与权重的乘积
            grad_weight_product = torch.abs(param.grad * param.data)
            grad_weight_product_list.append(grad_weight_product)

    # 将所有的乘积值合并到一个张量中
    all_product_values = torch.cat([torch.flatten(x) for x in grad_weight_product_list])
    # 计算需要修剪的阈值
    threshold = np.percentile(all_product_values.cpu().detach().numpy(), pruning_rate)

    # 对权重进行修剪
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 创建一个掩码，表示哪些权重应该保留
            mask = torch.where(torch.abs(param.grad * param.data) >= threshold, 1, 0)
            # 应用掩码
            param.data *= mask.float()

# 示例：使用50%的修剪率对一个PyTorch模型进行修剪
pruning_rate = 50
model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1))
input_tensor = torch.randn(1, 10)  # 创建一个随机输入张量
output_tensor = model(input_tensor)  # 前向传递
loss = torch.sum(output_tensor)  # 定义一个虚拟损失
loss.backward()                  # 执行反向传递以计算梯度
prune_by_gradient_weight_product(model, pruning_rate)  # 对模型进行修剪


# for name, param in model.named_parameters():
#     print(name, param.data)
