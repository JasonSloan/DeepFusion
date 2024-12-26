import numpy as np

def nms(boxes, scores, threshold):
    """
    非极大值抑制算法
    :param boxes: 边界框坐标列表，格式为 [x1, y1, x2, y2]
    :param scores: 边界框得分列表
    :param threshold: 重叠阈值
    :return: 保留的边界框的索引列表
    """
    # 根据边界框得分从高到低排序
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        # 保留得分最高的边界框
        i = indices[0]
        keep.append(i)
        # 计算当前边界框与其他边界框的重叠面积
        overlaps = np.zeros_like(indices[1:], dtype=np.float32)
        for j, k in enumerate(indices[1:]):
            box_i = boxes[i]
            box_j = boxes[k]
            x1 = max(box_i[0], box_j[0])
            y1 = max(box_i[1], box_j[1])
            x2 = min(box_i[2], box_j[2])
            y2 = min(box_i[3], box_j[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            overlap = w * h / ((box_i[2] - box_i[0]) * (box_i[3] - box_i[1]) + (box_j[2] - box_j[0]) * (box_j[3] - box_j[1]) - w * h)
            overlaps[j] = overlap
        # 保留重叠面积小于阈值的边界框
        indices = indices[1:][overlaps < threshold]
    return keep


if __name__ == '__main__':
    boxes = [
        [100, 100, 200, 200],
        [120, 110, 220, 210],
        [300, 320, 400, 400],
        [180, 100, 300, 180]
    ]
    scores = [0.9, 0.8, 0.7, 0.6]
    out = nms(boxes, scores, 0.5)
    print(out)




