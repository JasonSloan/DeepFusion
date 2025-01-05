# YOLOv5基于BN层的剪枝

Reference完整代码: https://github.com/midasklr/yolov5prune

## 1. 整体思路

正常训练---->

使用正常训练的权重进行稀疏训练---->基于稀疏训练的权重剪枝---->将剪枝后的权重进行finetune---->

将finetune后的权重进行稀疏训练---->基于稀疏训练的权重剪枝---->将剪枝后的权重进行finetune---->

将finetune后的权重进行稀疏训练---->基于稀疏训练的权重剪枝---->将剪枝后的权重进行finetune---->

......

![](assets/process.png)

## 2. 稀疏训练思路

在loss.backward()之后, optimizer.step()之前加上一段给BN层参数增加L1正则项梯度的代码

注意:所有的Res Unit模块的BN层不进行稀疏化训练, 所以也不增加L1正则项梯度

```python
srtmp = opt.sr * (1 - 0.9 * epoch / epochs)  # 线性衰减的L1正则化系数
if opt.st:
    ignore_bn_list = []
    for k, m in model.named_modules():
        if isinstance(m, Bottleneck):
             # 只有Bottleneck模块(对应于网络结构图中的Res Unit)中才做add操作, 所以不能剪
            if m.add:       
                # C3模块中的第一个卷积层
                ignore_bn_list.append(k.rsplit(".", 2)[0] + ".cv1.bn")
                # ignore_bn_list.append(k + '.cv1.bn')  # 源代码中该行应注释, 对应于prune.py中该行也应注释 	
                # C3模块中的BottleNeck模块中的第二个卷积层
                ignore_bn_list.append(k + '.cv2.bn')  
                if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
                    m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))  # L1
```

## 3. 剪枝思路

取出所有增加L1正则梯度项的BN层---->将这些BN层的所有.weight.data(也就是gamma的值)取出来放在一个tensor中---->将tensor按照大小排序---->为了避免剪枝的时候将某一整个BN层全部剪掉, 需要计算可剪枝的最大阈值, 计算方法为计算每一个BN层的weight的最大值, 然后在所有BN层的weight的最大值中选一个最小的值, 剪枝的阈值不能超过该值即可保证不会将某一层全部剪掉---->计算按照上一步方式计算的阈值剪枝的话最大能剪枝的比例---->计算按照指定比例剪枝的话剪枝的阈值(指定比例不能超过最大比例)---->遍历模型的每一层, 获得BN层, 按照上一步计算的阈值计算BN层的掩码---->将掩码与BN层的weight(gamma值)想乘, 与BN层的weight(beta值)想乘---->重新搭建网络, 此时的网络结构每一层的输入通道数与输出通道数应该与剪枝后的输入通道数输出通道数相同(比如原来是搭建一个cin=16, cout=32的卷积, 现在要搭建一个cin=14, cout=20的卷积)---->遍历搭建好的网络, 将稀疏训练中训练好的参数填入网络

## 4. 注意事项

 所有的Res Unit模块的BN层不进行稀疏化训练(因为有add操作, 所以要保证Res Unit前后的特征图的通道数相等)
