from torch import nn
import torch
import onnx
from onnx import helper
import numpy as np
import cv2

model = onnx.load("rrdb.onnx")
# print(model)	# 最好先把模型打印一下，看看每个节点的结构，或者使用netron看看模型每个节点的结构


# 使用onnx将预处理和后处理代码变成网络结构添加到网络中
# step0:构建预处理和后处理网络并转成onnx
class Postprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, 0, 1)[0]
        x = x.permute(1, 2, 0)
        x = x[:, :, [2, 1, 0]]
        x = x * 255.0
        return x

# class Preprocess(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         # Assuming x has the shape [H, W, C] and channel order BGR
#         x = x[:, :, :, [2, 1, 0]]
#         x = x.permute(0, 3, 1, 2)
#         x = x / 255.0  # Normalize
#         return x

def modify_onnx():
    """给onnx模型增加预处理和后处理部分"""
    # =====================Part1:给原模型增加后处理部分=====================
    # step1:构建后处理网络并转成onnx
    post = Postprocess().eval()
    dummy = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    dynamic = {
        "post": {2: "height", 3: "width"},
        "output": {2: "height", 3: "width"},
    }
    # dynamic = {
    #     "post": {0: "batch"},
    #     "output": {0: "batch"},
    # }
    torch.onnx.export(
        post, 
        (dummy), 
        "post.onnx",
        input_names=["post"], 
        output_names=["output"],
        opset_version=11,
        dynamic_axes = dynamic
        )

    # step2:加载后处理网络，将模型中的以output为输出的节点，修改为post_onnx的输入节点
    post_onnx = onnx.load("post.onnx")
    # 给所有的post-onnx网络的节点名字前面加上'post/'
    for item in post_onnx.graph.node:
        # 修改当前节点的名字
        item.name = f"post/{item.name}"
        # 修改当前节点的输入的名字
        for index in range(len(item.input)):
            item.input[index] = f"post/{item.input[index]}"
        # 修改当前节点的输出的名字
        for index in range(len(item.output)):
            item.output[index] = f"post/{item.output[index]}" 
    # 修改原模型的最后一层的输出节点名字改为post-onnx的输入节点的名字
    for item in model.graph.node:
        if item.name == "/conv_last/Conv":      # 这个需要到netron上查看到底叫啥名
            item.output[0] = "post/" + post_onnx.graph.input[0].name
            print("Change original model output to post model input successfully!")

    # setp3: 把post-onnx的node全部放到原模型的node中
    for item in post_onnx.graph.node:
        model.graph.node.append(item)    # 这里我看了model.graph.node这个转成列表后append不是在网络末尾追加吗，但是这个是将预处理加入到网络首部中，不应该是insert吗
    # 答：其实model.graph.node这个列表里的元素可以使完全乱序的，因为这个列表里的每个元素都标记好了他的输入是叫啥名，输出时叫啥名，所以无论在列表中顺序怎么乱，最终都能按照名字一一对应上

    # step4: 把post-onnx的输出名称作为原模型的输出名称
    output_name = "post/" + post_onnx.graph.output[0].name
    model.graph.output[0].CopyFrom(post_onnx.graph.output[0])
    model.graph.output[0].name = output_name

    # # =====================Part2:给原模型增加预处理部分=====================
    # # step1:构建预处理网络并转成onnx
    # pre = Preprocess().eval()
    # dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    # dynamic = {
    #     "pre": {1: "height", 2: "width"},
    #     "input": {1: "height", 2: "width"},
    # }
    # torch.onnx.export(
    #     pre, 
    #     (dummy), 
    #     "pre.onnx",
    #     input_names=["input"], 
    #     output_names=["pre"],
    #     opset_version=11,
    #     dynamic_axes = dynamic
    #     )
    
    # # step2:加载预处理网络，将模型中的以input为输入的节点，修改为pre_onnx的输出节点
    # pre_onnx = onnx.load("pre.onnx")
    # # 给所有的pre-onnx网络的节点名字前面加上'pre/'
    # for item in pre_onnx.graph.node:
    #     # 修改当前节点的名字
    #     item.name = f"pre/{item.name}"
    #     # 修改当前节点的输入的名字
    #     for index in range(len(item.input)):
    #         item.input[index] = f"pre/{item.input[index]}"
    #     # 修改当前节点的输出的名字
    #     for index in range(len(item.output)):
    #         item.output[index] = f"pre/{item.output[index]}"
    # # 修改原模型的第一层的输入节点名字改为pre-onnx的输出节点的名字
    # for item in model.graph.node:
    #     if item.name == "/conv_first/Conv":      # 这个需要到netron上查看到底叫啥名
    #         item.input[0] = "pre/" + pre_onnx.graph.output[0].name
    #         print("Change original model input to pre model output successfully!")
    
    # # step3: 把pre-onnx的node全部放到原模型的node中
    # for item in pre_onnx.graph.node:
    #     model.graph.node.append(item)    # 这里我看了model.graph.node这个转成列表后append不是在网络末尾追加吗，但是这个是将预处理加入到网络首部中，不应该是insert吗
    # # 答：其实model.graph.node这个列表里的元素可以使完全乱序的，因为这个列表里的每个元素都标记好了他的输入是叫啥名，输出时叫啥名，所以无论在列表中顺序怎么乱，最终都能按照名字一一对应上

    # # step4: 把pre-onnx的输入名称作为原模型的输入名称
    # input_name = "pre/" + pre_onnx.graph.input[0].name
    # model.graph.input[0].CopyFrom(pre_onnx.graph.input[0])
    # model.graph.input[0].name = input_name

    onnx.save(model, "../workspace/new.onnx")
    print("Done!")

def verify_result():
    """将原模型不做后处理推理结果使用pickle保存，然后使用自定义的后处理网络加载推理验证结果是否正确"""
    import pickle
    with open("result.pkl", mode="rb") as f:
        res = pickle.load(f)
    post = Postprocess()
    res = post(res)
    res = res.to(torch.uint8)
    res = res.cpu().numpy()
    cv2.imwrite("post_result.jpg", res);
    
if __name__ == '__main__':
    # verify_result()
    modify_onnx()
