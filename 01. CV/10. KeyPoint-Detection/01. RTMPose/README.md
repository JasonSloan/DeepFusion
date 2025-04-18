![](assets/1.jpg)

RTMPose将关键点回归任务变为分类任务, 模型最终分别预测x坐标的[batch_size, number_keypoints, number_bins]以及y坐标的[batch_size, number_keypoints, number_bins]的两个矩阵

例如假如我们要做个68点关键点检测任务, 原图大小为500x500, bins数为256:

骨干网络: CSPNeXt, 特征提取, 没什么特别的

7x7Conv: 没什么特别的, 作者试验出来发现在后面采取更大卷积核能提高mAP

FC: 7x7卷积后, 得到的feature map数量为[H, W, 68], 然后要进行flatten操作, 

变为[HxW, 68], 再transpose变为[68, HxW], 然后做一个全连接, 变为[68, 256]

GAU: Gate Attention Unit, Transformer self-attention注意力机制的变体

推理阶段: 循环遍历这68个点, 然后将这256个位置的概率乘以bin的位置求和, 再乘以原图大小即为预测出的关键点在原图的坐标

$$\hat{x} = \sum_{i=0}^{255} P(i) \cdot \left( \frac{i}{256} \times W \right)$$

$$\hat{y} = \sum_{i=0}^{255} P(i) \cdot \left( \frac{i}{256} \times H \right)$$

训练阶段: 假如第0号关键点的真实标注坐标为(100, 200), 那么100落在第100/500x256=51.2个bin处, 200落在第200/500x256=102.4个bin处, 将该真实点变为一个高斯分布的软标签, 也就是说x坐标的第49 50 51 52处均有概率, y坐标的第101 102 103 104处均有概率, 然后和预测值进行损失计算
