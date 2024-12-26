import os

import torch
import torch.nn as nn
# from torchvision.models import resnet50
from resnet import resnet50
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


def t1():
    # for test
    resnet = resnet50()
    data = torch.randn(32, 3, 32, 32)
    result = resnet(data)
    print(result.shape)


def train(epochs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Loading models...")
    model = resnet50()
    print("Loading models successfully!")
    root_dir = "./datasets"
    os.makedirs(root_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(64, antialias=True)])
    datasets = CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(datasets, 4, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    batches = 0
    losses = 0
    print(f"Start Training for {epochs} epochs , {epochs * len(dataloader)} batches...")
    for epoch in range(epochs):
        for i, (imgs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            losses+=loss
            batches+=1
            if batches % 10 == 0:
                avg_loss = losses / batches
                print(f"epoch : {epoch} \t batches : {batches} \t average_loss : {avg_loss:.3f}")






if __name__ == '__main__':
    # t1()
    epochs = 10
    train(epochs)
