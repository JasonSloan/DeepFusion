import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

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

def train_with_pruning(model, dataloader, criterion, optimizer, device='cpu', num_epochs=10, pruning_rate=0.5):
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

        # 在每个 epoch 结束后进行剪枝
        prune_network(model, pruning_rate, method="global") # <================================== just prune the weights ot 0 but still allow them to grow back by optimizer.step()

    return model

big_model = BigModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(big_model.parameters(), lr=1e-3)
big_model = train_with_pruning(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=10, pruning_rate=0.1)

# 保存训练好的模型
torch.save(big_model.state_dict(), "trained_with_pruning_model.pth")
