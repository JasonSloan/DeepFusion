import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
from torchvision.models import resnet18
import torch_pruning as tp


def create_dataloader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
        )
    return dataloader

def prune(device):
    dataloader = create_dataloader()
    model = resnet18(pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    # (剪枝-->finetune)循环的轮数
    iterative_steps = 5
    # GroupTaylorImportance需要用到
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    example_targets = torch.ones(1, dtype=torch.long).to(device)
    example_targets[0] = 1
    
    # 1. 重要程度的度量指标
    imp = tp.importance.GroupTaylorImportance() # or GroupNormImportance(p=2), GroupHessianImportance(), etc.
    
    # 2. 选择剪枝忽略的层(本例子中最后一个全连接层忽略); 最简单的方式是哪一层报错了就把哪一层添加到ignored_layers中
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) 
            
    # 3. 初始化pruner
    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps, # 迭代剪枝，设为1则一次性完成剪枝
        pruning_ratio=0.3, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        # global_pruning是将全局的BN层的参数放在一起排序, 选取阈值, 决定剪哪个; 非global_pruning是层内部的BN层的参数放在一起排序, 选取阈值, 决定剪哪个
        global_pruning=True,    
        ignored_layers=ignored_layers,
    )

    # 3. 计算模型初始参数量和计算量
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    if isinstance(imp, tp.importance.GroupTaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        outputs = model(example_inputs)
        loss = loss_fn(outputs, example_targets)
        loss.backward() # before pruner.step()
        
    # 4. Pruning-Finetuning的循环
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("Params: {:.2f} M => {:.2f} M".format(base_nparams / 1e6, nparams / 1e6))
        print("MACs: {:.2f} G => {:.2f} G".format(base_macs / 1e9, macs / 1e9)) 
        finetune_model(model, optimizer, loss_fn, dataloader, epochs=1)
        
    # 5. 保存模型
    model.zero_grad() # Remove gradients
    torch.save(model, 'model.pth') # without .state_dict
    dummies = torch.randn(1, 3, 224, 224).to(device)
    model = torch.load('model.pth')
    torch.onnx.export(
        model, 
        dummies,
        'model.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
        
def finetune_model(model, optimizer, loss_fn, dataloader, epochs):
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for batch in pbar:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_description(desc=f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune(device)