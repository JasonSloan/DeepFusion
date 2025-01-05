import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 1. 训练基础的大网络
class BigModel(nn.Module):
    def __init__(self):
        super(BigModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 准备MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train(model, dataloader, criterion, optimizer, device='cpu', num_epochs=10):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    return model

big_model = BigModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(big_model.parameters(), lr=1e-3)
big_model = train(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=2)

# 保存训练好的大网络
torch.save(big_model.state_dict(), "big_model.pth")

# 2. 修剪大网络为小网络  <==================================
def prune_network(model, pruning_rate=0.5, method="global"):
    for name, param in model.named_parameters():
        if "weight" in name:
            tensor = param.data.cpu().numpy()
            if method == "global":
                threshold = np.percentile(abs(tensor), pruning_rate * 100)
            else:  # local pruning
                threshold = np.percentile(abs(tensor), pruning_rate * 100, axis=1, keepdims=True)
            mask = abs(tensor) > threshold
            param.data = torch.FloatTensor(tensor * mask.astype(float)).to(param.device)


big_model.load_state_dict(torch.load("big_model.pth"))
prune_network(big_model, pruning_rate=0.5, method="global") # <==================================

# 保存修剪后的模型
torch.save(big_model.state_dict(), "pruned_model.pth")

# 3. 以低的学习率做微调
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(big_model.parameters(), lr=1e-4) # <==================================
finetuned_model = train(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=10)

# 保存微调后的模型
torch.save(finetuned_model.state_dict(), "finetuned_pruned_model.pth")


# Epoch 1, Loss: 0.2022465198550985
# Epoch 2, Loss: 0.08503768096334421
# Epoch 1, Loss: 0.03288614955859935
# Epoch 2, Loss: 0.021574671817958347
# Epoch 3, Loss: 0.015933904873507806