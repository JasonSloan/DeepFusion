# reference: https://github.com/yzfzzz/Stereo-Detection
import math
import os
import os.path as osp

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from utils import *
from configs import *


class SGBMDepthEstimator:
    """本类实现的功能: 
            1. 检查本地连接的摄像头可用情况, 并将可用摄像头的图像imshow出来(由use_video控制)
            2. 使用sgbm算法估计深度, 注意需要改configs.py的标定参数, 以及确保左右摄像头的传递顺序不要反
            3. 在sgbm估计深度期间, 会将深度图imshow出来, 摁's'会保存当前帧的校正后的左右图到本地'rectified-images'文件夹下
               点击深度图, 会在终端打印该图像位置的深度信息
            4. 注意sgbm算法要取得好的效果, 最重要的是调参, 本参数不一定会适用别的双目相机
    """

    def __init__(
        self,
        use_video=True,
        minDisparity=1,
        numDisparities=64,
        blockSize=5,
        P1=None,
        P2=None,
        disp12MaxDiff=-1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_HH,
        dynamic_tune=False,
    ):
        """_summary_

        Args:
            use_video (_type_): True为使用双目相机录制保存到本地的视频, false为使用相机实时视频
            minDisparity (_type_): 两张图之间同一个block沿着对极线搜索时视差像素的最低值, 单位: 像素(偏移小于该像素的将不会被考虑), 需要大于0
            numDisparities (_type_): 两张图之间同一个block沿着对极线搜索时视差像素的最大值减最小值, 单位: 像素(偏移大于该像素的将不会被考虑),
                                    需要被16整除, 越大精度越好, 速度越慢
            blockSize (_type_): 两张图之间匹配时block的大小, 奇数, 建议3...11之间
            P1 (_type_): 平滑系数, 增加P1使算法对视差变化的惩罚更大, 从而导致视差图更平滑, 相邻像素之间的视差跳跃更小,
                        推荐值8 * img_channels * blockSize * blockSize
            P2 (_type_): 平滑系数, 控制相邻像素之间较大视差跳跃的惩罚; 增加P2意味着算法强烈阻止视差的突然变化, 强制实现更平滑的过渡,
                        尤其是跨对象边界, 推荐值32 * img_channels * blockSize * blockSize
            disp12MaxDiff (_type_): 左右视差检查（视差一致性检查）中允许的最大差异。负值会禁用检查; 通常，该参数确保左右图像生成的视差图彼此一致。
                                    典型值可能类似于 1 或 2
            preFilterCap (_type_): 用于预过滤图像的截断值。它在视差计算之前限制像素值以减少图像噪声的影响; 该值控制在视差计算之前如何剪切图像强度,
                                    具体为计算x方向的梯度, x方向梯度超过[-preFilterCap, preFilterCap]区间的会被裁剪。通常，它设置得较高（例如 31 或 63),
                                    目的是标准化输入图像的强度范围，以便极端的强度变化不会对立体匹配产生负面影响。立体算法通常依赖于图像区域的相似性，
                                    亮度或极端像素值的太大变化可能会混淆匹配过程
            uniquenessRatio (_type_): 为10意味着所选视差必须比下一个最佳候选视差好至少 10%。这有助于通过确保视差唯一性来减少噪音并提高准确性; 推荐值5-15
            speckleWindowSize (_type_): 为100意味着具有相似视差值且少于 100 个像素的区域将被视为噪声并被删除。这有助于减少随机的小视差伪影
            speckleRange (_type_): 如果一个区域内最大视差变化超过该值, 那么该区域被视为噪声, 会被去除。如果进行散斑过滤, 将参数设置为正值, 它将隐式乘以16。建议值1或2
            mode (_type_): sgbm算法选择模式, 以速度由快到慢为: STEREO_SGBM_MODE_SGBM_3WAY、STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
            dynamic_tune (_type_): 可视化调参, 可视化调参只显示numDisparities和blockSize两个参数
        """
        self.is_bar_set = False
        self.use_video = use_video
        self.dynamic_tune = dynamic_tune
        if not use_video:
            self.list_available_cameras()
        self.init_sgbm(
            minDisparity,
            numDisparities,
            blockSize,
            P1,
            P2,
            disp12MaxDiff,
            preFilterCap,
            uniquenessRatio,
            speckleWindowSize,
            speckleRange,
            mode,
            dynamic_tune,
        )

    def list_available_cameras(self, max_range=10, verbose=True, show=True):
        available_devices = []
        for camera_idx in range(max_range):
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                available_devices.append(camera_idx)
                if show:
                    ret, _ = cap.read()
                    while ret:
                        ret, frame = cap.read()
                        frame = cv2.putText(
                            frame,
                            f"camera {camera_idx}",
                            (0, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow(f"camera {camera_idx}", frame)
                        if cv2.waitKey(1) == ord("q"):
                            cv2.destroyAllWindows()
                            break
                cap.release()
        if verbose:
            print(f"Available camera devices are: {available_devices}")

    def init_windows(self):
        self.left_frame_win_name = "left frame"         # 左相机的原图
        self.right_frame_win_name = "right frame"       # 右相机的原图
        self.disp_color_win_name = "disparity color"    # 渲染出的深度图的彩色图
        cv2.namedWindow(self.left_frame_win_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.right_frame_win_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.disp_color_win_name, cv2.WINDOW_AUTOSIZE)

    def init_sgbm(
        self,
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        disp12MaxDiff,
        preFilterCap,
        uniquenessRatio,
        speckleWindowSize,
        speckleRange,
        mode,
        dynamic_tune,
    ):
        self.init_windows()
        self.create_stereo_sgbm(
            minDisparity,
            numDisparities,
            blockSize,
            P1,
            P2,
            disp12MaxDiff,
            preFilterCap,
            uniquenessRatio,
            speckleWindowSize,
            speckleRange,
            mode,
            dynamic_tune,
        )

    def create_stereo_sgbm(
        self,
        minDisparity=1,
        numDisparities=64,
        blockSize=5,
        P1=None,
        P2=None,
        disp12MaxDiff=-1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_HH,
        dynamic_tune=False,
    ):
        img_channels = 3
        if dynamic_tune:
            if not self.is_bar_set:
                cv2.createTrackbar(
                    "numDisparities", self.disp_color_win_name, 16, 160, lambda x: None
                )  # 可视化调参
                numDisparities = cv2.getTrackbarPos(
                    "numDisparities", self.disp_color_win_name
                )
                numDisparities = numDisparities // 16 * 16  # 必须被16整除
                cv2.createTrackbar(
                    "blockSize", self.disp_color_win_name, 3, 11, lambda x: None
                )  # 可视化调参
                self.is_bar_set = True
            blockSize = cv2.getTrackbarPos("blockSize", self.disp_color_win_name)
            blockSize = blockSize if blockSize % 2 else blockSize + 1  # 必须是奇数
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=(P1 if P1 is not None else 8 * img_channels * blockSize * blockSize),
            P2=(P2 if P2 is not None else 32 * img_channels * blockSize * blockSize),
            disp12MaxDiff=disp12MaxDiff,
            preFilterCap=preFilterCap,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            mode=mode,
        )

    def create_mapping_table(self, caps):
        left_camera_matrix = camera_params["left"]["camera_matrix"]
        left_distortion = camera_params["left"]["distortion"]
        right_camera_matrix = camera_params["right"]["camera_matrix"]
        right_distortion = camera_params["right"]["distortion"]
        R = camera_params["R"]
        T = camera_params["T"]
        self.two_usbs = len(caps) == 2
        width = (
            int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            if self.two_usbs
            else int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
        )
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = [int(width), int(height)]
        R1, R2, P1, P2, self.Q, validPixROI1, validPixROI2 = (
            cv2.stereoRectify(  # 为了图像校正的固定代码不用管
                left_camera_matrix,
                left_distortion,
                right_camera_matrix,
                right_distortion,
                self.size,
                R,
                T,
            )
        )
        self.left_map1, self.left_map2 = (
            cv2.initUndistortRectifyMap(  # 为了图像校正的固定代码不用管
                left_camera_matrix, left_distortion, R1, P1, self.size, cv2.CV_16SC2
            )
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            right_camera_matrix, right_distortion, R2, P2, self.size, cv2.CV_16SC2
        )

    @staticmethod  # 鼠标回调函数
    def onmouse_pick_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            threeD = param
            print(f"Pixel cordinates: x = {x:0d}, y = {y:0d}")  # 像素坐标
            print(
                f"World cordinates: "  # 世界坐标
                f"x = {threeD[y][x][0] / 1000.0:.2f}m, "
                f"y = {threeD[y][x][1] / 1000.0:.2f}m, "
                f"z = {threeD[y][x][2] / 1000.0:.2f}m"
            )
            distance = math.sqrt(
                threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2
            )
            distance = distance / 1000.0
            print(f"Distance: {distance}m\n")

    def save_rectified(
        self, imgL, imgR, imgL_rectified, imgR_rectified, save_dir="rectified-images"
    ):
        os.makedirs(save_dir, exist_ok=True)
        cat_img_org = np.concatenate([imgL, imgR], 1)
        cat_img_rectified = np.concatenate([imgL_rectified, imgR_rectified], 1)
        img_shape = cat_img_org.shape
        for i in range(0, img_shape[0], 40):
            cv2.line(
                cat_img_org,
                pt1=(0, i),
                pt2=(img_shape[1], i),
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.line(
                cat_img_rectified,
                pt1=(0, i),
                pt2=(img_shape[1], i),
                color=(0, 255, 0),
                thickness=2,
            )
        save_path_org = osp.join(save_dir, f"{len(os.listdir(save_dir)) + 1}-org.jpg")
        save_path_rectified = osp.join(
            save_dir, f"{len(os.listdir(save_dir)) / 2 + 1}-rectified.jpg"
        )
        cv2.imwrite(save_path_org, cat_img_org)
        cv2.imwrite(save_path_rectified, cat_img_rectified)

    def estimate_depth(self, source):
        """_summary_

        Args:
            source (_type_, optional): 本地保存的视频或者双目相机的序号
        """
        caps = []
        sources = source if isinstance(source, list) else [source]
        assert len(sources) == 1 or len(sources) == 2, "wrong source"
        for s in sources:
            cap = cv2.VideoCapture(s)
            caps.append(cap)
        self.create_mapping_table(caps)  # 创建图像校正映射表
        if self.two_usbs:
            print(
                colorstr(
                    "yellow",
                    'you are using two-usb stereo camera, remember to put the left camera index at front when passing the "source" parameter',
                )
            )
        width = self.size[0]
        while True:
            frame = np.concatenate([cap.read()[1] for cap in caps], 1)
            imgL = frame[:, :width]
            imgR = frame[:, width:]
            # 图像校正
            imgL_rectified = cv2.remap(
                imgL, self.left_map1, self.left_map2, cv2.INTER_LINEAR
            )
            imgR_rectified = cv2.remap(
                imgR, self.right_map1, self.right_map2, cv2.INTER_LINEAR
            )
            # 计算视差
            disparity = self.stereo.compute(imgL_rectified, imgR_rectified)
            if self.dynamic_tune:
                self.create_stereo_sgbm(dynamic_tune=True)
            # 渲染彩色深度图
            disp_normed = cv2.normalize(
                disparity,
                disparity,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            disp_color = cv2.applyColorMap(disp_normed, 2)
            # 计算三维坐标数据值
            threeD = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)
            threeD = threeD * 16  # 计算出的threeD，需要乘以16，才等于现实中的距离
            # 设置鼠标回调事件
            cv2.setMouseCallback(
                self.disp_color_win_name, self.onmouse_pick_points, threeD
            )
            cv2.imshow(self.left_frame_win_name, imgL)
            cv2.imshow(self.right_frame_win_name, imgR)
            cv2.imshow(self.disp_color_win_name, disp_color)
            key = cv2.waitKey(1)
            if key == ord("s"):  # save joint rectified image
                self.save_rectified(imgL, imgR, imgL_rectified, imgR_rectified)
            if key == ord("q"):
                break
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 如果是使用本地文件, use_video为True, 如果要可视化调参dynamic_tune为True
    estimator = SGBMDepthEstimator(use_video=True, dynamic_tune=True)
    estimator.estimate_depth(source='stereo-video/car.avi') # car.avi链接 https://pan.baidu.com/s/1O8JA6I7BxRDycs-pGQSUag 提取码: rjaf 
    # # 如果是使用usb摄像头, use_video为True, 如果要可视化调参dynamic_tune为True
    # estimator = SGBMDepthEstimator(use_video=False, dynamic_tune=True)
    # estimator.estimate_depth(source=[0]) # 双目相机的摄像头序号
