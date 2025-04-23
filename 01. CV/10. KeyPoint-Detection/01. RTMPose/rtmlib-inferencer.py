import cv2
from rtmlib import Wholebody, draw_skeleton

img = cv2.imread('inputs/images/2.jpg')
wholebody = Wholebody(
    to_openpose=False,
    mode='balanced',        # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
    backend='onnxruntime',  # opencv, onnxruntime, openvino
    device='cpu'            # cpu, cuda, mps
)
keypoints, scores = wholebody(img)
keypoints = keypoints
scores = scores
img_show = img.copy()
img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)
cv2.imwrite('outputs/2.jpg', img_show)

# wholebody = Wholebody(
#     pose='../mmdeploy_model/rtmpose/end2end.onnx',  # 脸部关键点检测权重
#     pose_input_size=(256, 256),
#     to_openpose=False,
#     mode='balanced',        # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
#     backend='onnxruntime',  # opencv, onnxruntime, openvino
#     device='cpu'            # cpu, cuda, mps
# )
# keypoints, scores = wholebody(img)
# keypoints = keypoints[0] # 一次一张图
# scores = scores[0]       # 一次一张图
# img_show = img.copy()
# for i, (kpt, score) in enumerate(zip(keypoints, scores)):
#     kpt = kpt.astype(int)
#     cv2.circle(img_show, (kpt[0], kpt[1]), 2, (0, 255, 0), -1)
# cv2.imwrite('outputs/2.jpg', img_show)



