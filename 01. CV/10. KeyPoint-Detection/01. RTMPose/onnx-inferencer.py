import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
    

class RTMPoseOnnxInferencer:
    
    OUTER_LIP_INDICES = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]    # 嘴唇外轮廓
    INNER_LIP_INDICES = [96, 97, 98, 99, 100, 101, 102, 103]                # 嘴唇内轮廓
    INNER_UPPER_LIP_INDICES = [97, 98, 99]                                  # 嘴唇内轮廓上半部分
    INNER_LOWER_LIP_INDICES = [101, 102, 103]                               # 嘴唇内轮廓下半部分
    INNER_CORNER_LIP_INDICES = [96, 100]                                    # 嘴唇内轮廓角点
    LEFT_UPPER_EYE_INDICES = [67, 68, 69]                                   # 左眼上半部分
    LEFT_LOWER_EYE_INDICES = [71, 72, 73]                                   # 左眼下半部分
    LEFT_CORNER_EYE_INDICES = [66, 70]                                      # 左眼角点
    RIGHT_UPPER_EYE_INDICES = [76, 77, 78]                                  # 右眼上半部分
    RIGHT_LOWER_EYE_INDICES = [80, 81, 82]                                  # 右眼下半部分
    RIGHT_CORNER_EYE_INDICES = [75, 79]                                     # 右眼角点
    
    def __init__(self, onnx_model: str, thre: float = 0.6,
                 mean: tuple = (123.675, 116.28, 103.53), std: tuple = (58.395, 57.12, 57.375), 
                 simcc_split_ratio: int = 2, backend: str = 'onnxruntime', device: str = 'cuda'):
        """RTMPose ONNX推理器初始化函数.

        Args:
            onnx_model (str): ONNX模型路径.
            model_input_size (tuple): 模型输入大小.
            mean (tuple): 均值.
            std (tuple): 标准差.
            backend (str): 后端类型, 只支持'onnxruntime'.
            device (str): 设备类型, 如'cpu'或'cuda'.
        """
        self.thre = thre
        self.device = device
        self.mean = np.array(mean)
        self.std = np.array(std)
        assert backend == 'onnxruntime', 'Only onnxruntime backend is supported.'
        
        avail_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' not in avail_providers:
            print('CUDAExecutionProvider is not available, using CPUExecutionProvider instead.')
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_dtype = self.session.get_inputs()[0].type
        self.model_input_size = (self.input_shape[3], self.input_shape[2])
        self.simcc_split_ratio = simcc_split_ratio
        print(f"Model input name: {self.input_name}, shape: {self.input_shape}, dtype: {self.input_dtype}")

    def _prepare(self, inputs, save_dir):
        """
        准备输入数据和保存目录.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        is_video = False
        if inputs.is_file():
            if inputs.suffix in ['.jpg', '.png', '.jpeg']:
                imgs = [inputs]
                num_imgs = 1
            if inputs.suffix in ['.mp4', '.avi', '.mov']:
                is_video = True
                imgs = cv2.VideoCapture(str(inputs))
                num_imgs = int(imgs.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(imgs.get(cv2.CAP_PROP_FPS))
                frame_height = int(imgs.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_width = int(imgs.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.writer = cv2.VideoWriter(
                    save_dir / inputs.name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
        else:
            imgs = sorted(inputs.glob('*.jpg'))
            num_imgs = len(imgs)

        return imgs, num_imgs, is_video
    
    def _preprocess(self, img):
        """
        预处理图片
        """
        input_dict = {}
        img_h, img_w, _ = img.shape
        input_w, input_h = self.model_input_size
        ratio = max(img_h / input_h, img_w / input_w)
        new_w = int(img_w / ratio)
        new_h = int(img_h / ratio)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_wl = (input_w - new_w) // 2
        pad_wr = (input_w - new_w) - pad_wl
        pad_ht = (input_h - new_h) // 2
        pad_hb = (input_h - new_h) - pad_ht
        resized_img = cv2.copyMakeBorder(resized_img, pad_ht, pad_hb, pad_wl, pad_wr,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        std_img = (resized_img - self.mean) / self.std
        transposed_img = std_img.transpose(2, 0, 1)
        transposed_img = np.ascontiguousarray(transposed_img, dtype=np.float32)
        input = transposed_img[None, :, :, :]
        
        input_dict['input'] = input
        input_dict['ratio'] = ratio
        input_dict['pad_wl'] = pad_wl
        input_dict['pad_ht'] = pad_ht
        
        return input_dict

    def _infer(self, input):
        sess_input = {self.session.get_inputs()[0].name: input}
        sess_output = []
        for out in self.session.get_outputs():
            sess_output.append(out.name)

        outputs = self.session.run(sess_output, sess_input)
        return outputs
    
    def _postprocess(self, output, input_dict):
        """
        后处理模型输出, 主要是将simcc的bins索引还原为坐标
        """
        simcc_x, simcc_y = output
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
        x_locs = np.argmax(simcc_x, axis=1) # 获得概率最大的bin索引
        y_locs = np.argmax(simcc_y, axis=1) # 获得概率最大的bin索引
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)
        scores = 0.5 * (max_val_x + max_val_y) # 将x和y的平均值作为最终的score
        locs = locs.reshape(N, K, 2)
        scores = scores.reshape(N, K)
        
        keypoints = locs / self.simcc_split_ratio # 将在bins的索引还原为像素值(bins数量比像素值多, 因此可做到亚像素级别)
        keypoints = (keypoints - np.array([input_dict['pad_wl'], input_dict['pad_ht']])) * input_dict['ratio']
        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)
        
        return keypoints, scores

    def _save(self, img, img_name, keypoints, scores, save_dir):
        for indice, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < self.thre: # 这里的score实际上是未做softmax处理的logits, 此处认为小于0的点为无效点
                continue
            # if indice in RTMPoseOnnxInferencer.INNER_LOWER_LIP_INDICES:
            cv2.circle(img, kpt.astype(np.int32), 1, (0, 255, 0), -1)

        if img_name is not None:
            cv2.imwrite(save_dir / img_name, img)
        else:
            self.writer.write(img)

    def __call__(self, inputs, save_dir):
        inputs = Path(inputs)
        save_dir = Path(save_dir)
        imgs, num_imgs, is_video = self._prepare(inputs, save_dir)

        for img_indice in tqdm(range(num_imgs)):
            img = imgs.read()[1] if is_video else cv2.imread(imgs[img_indice])
            input_dict = self._preprocess(img)
            output = self._infer(input_dict['input'])
            keypoints, scores = self._postprocess(output, input_dict)
            img_name = None if is_video else imgs[img_indice].name
            self._save(img, img_name, keypoints, scores, save_dir)

        print(f'Results saved in {save_dir}')


if __name__ == '__main__':
    """
    基于RTMPose使用SimCC思想的ONNX推理器, 这里验证了face6模型的推理结果
    face6模型权重: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth
    模型权重获取方法, 请参考: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#-model-zoo-
    将pt模型转换为onnx模型, 请参考: https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/02-how-to-run/convert_model.md
    模型转换: 
    python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py \
    mmpose/configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py \
    mmpose/weights/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth \
    mmpose/inputs/images/2.jpg \
    --work-dir mmdeploy_model/rtmpose \
    --dump-info
    
    NOTE: 这里没有人脸目标检测, 只有人脸关键点检测, 所以如果输入不是只包含人脸区域, 那么结果会不正确
    本代码只是为了对RTMPose以及SimCC到底做了什么有个更清晰的了解, 进而方便部署C++推理代码
    如果想要测试人脸检测+人脸关键点检测的完整pipeline, 请使用mmpose-inferencer.py代码
    """
    inferencer = RTMPoseOnnxInferencer(onnx_model='../mmdeploy_model/rtmpose/end2end.onnx')
    inferencer(inputs='inputs/images', save_dir='outputs/images')
    