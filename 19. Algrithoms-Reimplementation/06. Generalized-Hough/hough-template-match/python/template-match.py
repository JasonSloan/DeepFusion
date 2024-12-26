'''
Created on May 19, 2013

@author: vinnie
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from scipy.misc import imread
from imageio import imread
from skimage.feature import canny
from scipy.ndimage.filters import sobel
import cv2

# Good for the b/w test images used
MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50


def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    # np.arctan2接受两个数值，np.arctan接受一个数值
    # * 180 / np.pi : 弧度转角度
    gradient = np.arctan2(dy, dx) * 180 / np.pi

    return gradient


def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    # 边缘检测， egdes：bool值的图，位置为True代表是边缘，False代表不是
    edges = canny(image, low_threshold=MIN_CANNY_THRESHOLD,
                  high_threshold=MAX_CANNY_THRESHOLD)
    # 计算梯度方向，gradient：每个位置的值代表了该位置的梯度方向
    gradient = gradient_orientation(edges)

    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            # r_table的key是梯度方向，value是一个列表，列表中存储着是该梯度方向的点，这样的点可能有很多个
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table


def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD,
                  high_threshold=MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)

    accumulator = np.zeros(grayImage.shape)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            # 获得r_table中当前梯度角度下的向量集
            for r in r_table[gradient[i, j]]:
                # 将当前点恢复成中心点
                accum_i, accum_j = int(i + r[0]), int(j + r[1])
                # 对中心点进行投票
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[accum_i, accum_j] += 1

    return accumulator


def general_hough_closure(reference_image):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image
    
    Returns a function f, which takes a query image and returns the accumulator
    '''
    # 参考点选取模版图片的中心点
    referencePoint = (reference_image.shape[0] / 2, reference_image.shape[1] / 2)
    r_table = build_r_table(reference_image, referencePoint)

    def f(query_image):
        return accumulate_gradients(r_table, query_image)

    return f


def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]


def test_general_hough(gh, reference_image, query):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    query_image = cv2.imread(query, 0)
    accumulator = gh(query_image)

    plt.clf()
    plt.gray()

    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.title('Reference image')
    plt.imshow(reference_image)

    fig.add_subplot(2, 2, 2)
    plt.title('Query image')
    plt.imshow(query_image)

    fig.add_subplot(2, 2, 3)
    plt.title('Accumulator')
    plt.imshow(accumulator)

    fig.add_subplot(2, 2, 4)
    plt.title('Detection')
    plt.imshow(query_image)

    # top 5 results in red
    m = n_max(accumulator, 5)
    y_points = [pt[1][0] for pt in m]
    x_points = [pt[1][1] for pt in m]
    plt.scatter(x_points, y_points, marker='o', color='r')

    # top result in yellow
    i, j = np.unravel_index(accumulator.argmax(), accumulator.shape)
    plt.scatter([j], [i], marker='x', color='y')

    d, f = os.path.split(query)[0], os.path.splitext(os.path.split(query)[1])[0]
    plt.savefig(os.path.join(d, f + '_output.png'))

    return


def test():
    reference_image = cv2.imread("images/s.png", 0)
    detect_s = general_hough_closure(reference_image)
    test_general_hough(detect_s, reference_image, "images/s_test.png")

if __name__ == '__main__':
    test()
