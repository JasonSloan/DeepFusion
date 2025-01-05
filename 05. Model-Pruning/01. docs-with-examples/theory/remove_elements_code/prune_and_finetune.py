# prune and finetune

from train import BigModel, train
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. load a model and inspect it
model = BigModel()
model.load_state_dict(torch.load("big_model.pth"))

# 2. get the global threshold according to the l1norm
all_l1norm_values = []
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        l1norm_buffer_name = f"{name}_l1norm_buffer"
        l1norm = getattr(model, l1norm_buffer_name)         # 这里l1norm_buffer_name是动态变化的,不能直接通过model.l1norm_buffer_name来获取
        all_l1norm_values.append(l1norm)

threshold = torch.sort(all_l1norm_values[0])[0][int(len(all_l1norm_values[0]) * 0.5)] 

# 3. prune the conv based on the l1norm along axis = 0 for each weight tensor
conv1 = model.conv1 # torch.Size([32, 1, 3, 3])
conv2 = model.conv2 #            [16, 32,3, 3]
fc    = model.fc

conv1_l1norm_buffer = model.conv1_l1norm_buffer # 32
conv2_l1norm_buffer = model.conv2_l1norm_buffer # 16


# Top conv
keep_idxs = torch.where(conv1_l1norm_buffer >= threshold)[0]
k = len(keep_idxs)

# print(f"{k / len(conv1_l1norm_buffer) * 100}% kept")

conv1.weight.data = conv1.weight.data[keep_idxs]
conv1.bias.data   = conv1.bias.data[keep_idxs]
conv1_l1norm_buffer.data = conv1_l1norm_buffer.data[keep_idxs]
conv1.out_channels = k

# Bottom conv 
_, keep_idxs = torch.topk(conv2_l1norm_buffer, k)
conv2.weight.data = conv2.weight.data[:,keep_idxs] 
conv2.in_channels = k

# code snippest for inspecting the structure
for name, module in model.named_modules():
    print(name, module)
    pass

# Save the pruned model state_dict
torch.save(model.state_dict(), "pruned_model.pth")

# Set the input shape of the model
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "pruned_model.onnx")

#################################### FINE TUNE ######################################
# Prepare the MNIST dataset
model.load_state_dict(torch.load("pruned_model.pth"))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
big_model = train(model, train_loader, criterion, optimizer, device='cuda', num_epochs=3)

# Save the trained big network
torch.save(model.state_dict(), "pruned_model_after_finetune.pth")

# (/home/coeyliang/miniconda3) coeyliang_pruning> python pruneAndFinetune.py 
#  BigModel(
#   (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (fc): Linear(in_features=12544, out_features=10, bias=True)
# )
# conv1 Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# conv2 Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# fc Linear(in_features=12544, out_features=10, bias=True)
# Epoch 1, Loss: 0.12122726230745587
# Epoch 2, Loss: 0.0353821593767175
# Epoch 3, Loss: 0.022971595964452966



