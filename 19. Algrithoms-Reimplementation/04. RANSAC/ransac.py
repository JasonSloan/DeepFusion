import math
import random
import numpy as np

def ransac(points, max_dist_of_inner_points, inner_points_ratio_thre=0.8, 
           confidence=0.99, max_iters=10000, epsilon=1e-6):
    """使用RANSAC算法拟合直线

    Args:
        points (_type_): 被拟合的点集: List[(x1, y1), (x2, y2), (x3, y3)...(xn, yn)]
        max_dist_of_inner_points (_type_): 最大的被算作内点的点到直线的距离
        inner_points_ratio_thre (float, optional): 内点占比达到多少阈值即停止迭代 Defaults to 0.8.
        confidence (float, optional): 更新最大迭代次数的时置信度 Defaults to 0.99.
        max_iters (int, optional): 最大迭代次数, 初始值为10000. Defaults to 10000.
        epsilon (_type_, optional): 防止被除数等于0. Defaults to 1e-6.

    Returns:
        _type_: 拟合出的k和b的值, y = kx + b
    """
    # 这里假设只拟合直线, 拟合一条直线需要两对点
    num_points = 2
    # 当前迭代次数
    current_iter = 0
    best_k = 0
    best_b = 0
    # 最大内点占比, 初始为0
    max_inner_points_ratio = 0
    # max_iters初始为1万, 这个无所谓, 第一次迭代就会更新
    # 如果迭代次数小于最大迭代次数且最大内点占比没有达到阈值就继续迭代
    while current_iter < max_iters and max_inner_points_ratio < inner_points_ratio_thre:
        # 随机抽样2组点对
        sampled_points = random.sample(points, num_points)
        points1, points2 = sampled_points
        x1, y1 = points1
        x2, y2 = points2
        # 计算直线的k和b, y = kx + b
        k = (y2 - y1) / (x2 - x1 + epsilon)
        b = y1 - k * x1
        # 计算当前内点占比
        current_inner_points_ratio = compute_inner_points_ratio(k, b, points, max_dist_of_inner_points)
        # 如果当前内点占比大于最大内点占比, 更新最大内点占比和最大迭代次数
        if current_inner_points_ratio > max_inner_points_ratio:
            # 更新最大迭代次数
            max_iters = update_max_iters(current_inner_points_ratio, confidence, num_points, epsilon)
            max_inner_points_ratio = current_inner_points_ratio
            best_k = k
            best_b = b
            
        current_iter += 1 
        if current_iter % 10 == 0:
            print(f"current iter / max_iters: {current_iter} / {max_iters}\n"
                  f"current inner points ratio: {current_inner_points_ratio}\n"
                  f"best k: {best_k}, best b: {best_b}")         
    return best_k, best_b
    
def compute_inner_points_ratio(k, b, points, max_dist_of_inner_points):
    num_inners  = 0
    for point in points:
        x, y = point
        # 点到直线距离
        dist = abs(k * x - y + b) / math.sqrt(k**2 + 1)
        if dist < max_dist_of_inner_points:
            num_inners += 1
    inner_points_ratio = num_inners / len(points)
    return inner_points_ratio

def update_max_iters(current_inner_points_ratio, confidence, num_points, epsilon=1e-6):
    p = confidence
    ep = 1 - current_inner_points_ratio
    return math.log(1 - p) / (math.log(1 - (1 - ep)**num_points) + epsilon)
    


if __name__ == "__main__":
    # 假设要拟合的直线k=2, b=1, 真实内点占比0.3, 生成1000个点对
    k = 2
    b = 1
    inner_ratio = 0.3
    num_points = 1000
    xs = random.sample(range(-num_points, num_points), num_points)
    ys = []
    for i in range(num_points):
        # 内点
        if random.random() < inner_ratio:
            y = k * xs[i] + b 
        # 外点
        else:
            # 更一般的直线Ax + By + C = 0
            A = np.random.uniform(-1, 1, 1).item()
            B = np.random.uniform(-1, 1, 1).item()
            C = np.random.uniform(-1, 1, 1).item()
            y = (A * xs[i] + C) / (-B + 1e-6)
        ys.append(y)
    points = list(zip(xs, ys))
    k, b = ransac(points, max_dist_of_inner_points=1,
           inner_points_ratio_thre=0.8, confidence=0.99, 
           max_iters=10000, epsilon=1e-6)
    print("k:", k)
    print("b:", b)