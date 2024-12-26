import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def load_data(file_name=r'E:\AI\DeepBlue_AI_Course\Code\07_test_before_practice\canny_practice\pics\cat.png'):
    img = mpimg.imread(file_name)
    img = rgb2gray(img)
    return img


def visualize(img, format=None):
    plt.figure(figsize=(20, 40))
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    plt.imshow(img, format)
    plt.show()
