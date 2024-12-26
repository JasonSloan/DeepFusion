import utils
from canny_edge_detector import cannyEdgeDetector

if __name__ == '__main__':
    img = utils.load_data()
    detector = cannyEdgeDetector(img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17,
                                     weak_pixel=100)
    imgs_final = detector.detect()
    utils.visualize(imgs_final, 'gray')