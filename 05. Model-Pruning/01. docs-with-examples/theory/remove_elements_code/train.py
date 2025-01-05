# train phase

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 1. Train a large base network
class BigModel(nn.Module):
    def __init__(self):
        super(BigModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, 10)

        # Initialize l1norm as a parameter and register as buffer
        # 不需要求梯度,所以使用nn.Parameter进行包裹
        self.conv1_l1norm = nn.Parameter(torch.Tensor(32), requires_grad=False)
        self.conv2_l1norm = nn.Parameter(torch.Tensor(16), requires_grad=False)
        # self.register_buffer作用是,如果没有register_buffer,那么模型就不能to('cuda')
        # 此时self.conv1_l1norm和self.conv2_l1norm在cpu上,而其他参数可以到gpu上,数据不在同一设备
        # 而且没有self.register_buffer的话,torch.save不会将self.conv1_l1norm和self.conv2_l1norm
        # 保存在state_dict()中,因此也就load不出来
        self.register_buffer('conv1_l1norm_buffer', self.conv1_l1norm)
        self.register_buffer('conv2_l1norm_buffer', self.conv2_l1norm)
        

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        self.conv1_l1norm.data = torch.sum(torch.abs(self.conv1.weight.data), dim=(1, 2, 3))
        
        x = torch.relu(self.conv2(x))
        self.conv2_l1norm.data = torch.sum(torch.abs(self.conv2.weight.data), dim=(1, 2, 3))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Training function
def train(model, dataloader, criterion, optimizer, device='gpu', num_epochs=10):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(f"Loss: {running_loss / len(dataloader)}")
            
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    return model


if __name__ == "__main__":
    # Prepare the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    big_model = BigModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(big_model.parameters(), lr=1e-3)
    big_model = train(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=3)

    # Save the trained big network
    torch.save(big_model.state_dict(), "big_model.pth")
    
    # Set the input shape of the model
    dummy_input = torch.randn(1, 1, 28, 28).to("cuda")

    # Export the model to ONNX format
    torch.onnx.export(big_model, dummy_input, "big_model.onnx")

# (/home/coeyliang/miniconda3) coeyliang_pruning> python train.py 
# Epoch 1, Loss: 0.14482067066501939
# Epoch 2, Loss: 0.05070804020739657
# Epoch 3, Loss: 0.03378467213614771


