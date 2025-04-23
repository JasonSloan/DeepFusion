from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer
import warnings
warnings.filterwarnings('ignore')


class M_MMPoseInferencer:
    
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
    
    def __init__(self, aspect: str = 'face', thre: float = 0.6, device: int = 'cuda:0', 
                 judge_tired=True, yawn_thre: float = 0.5, eye_closed_thre: float = 0.2):
        """
        Args:
            aspect: str, default='face', 哪个方面的关键点检测, 可选为'face', 'body', 'hand'
            thre: float, default=0.6, 置信度阈值
            device: str, default='cuda:0', 设备类型, 如'cpu'或'cuda:0'
            judge_tired: bool, default=True, 是否进行疲劳判断(打哈欠, 眼睛闭合)
        """
        self.thre = thre
        self.judge_tired = judge_tired
        self.yawn_thre = yawn_thre
        self.eye_closed_thre = eye_closed_thre
        self.inferencer = MMPoseInferencer(aspect, device=device)
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
    def _draw_landmarks(self, img, landmarks, scores, color, threshold=0.6):
        for (x, y), score in zip(landmarks, scores):
            if score < threshold:
                continue
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
    def _save(self, img, img_name, keypoints, scores, save_dir):
        if not self.judge_tired:
            for indice, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score < self.thre: # 这里的score实际上是未做softmax处理的logits, 此处认为小于0的点为无效点
                    continue
                cv2.circle(img, kpt.astype(np.int32), 1, (0, 255, 0), -1)
        else:
            # 获取唇部对应关键点
            inner_upper_lip_keypoints = keypoints[self.INNER_UPPER_LIP_INDICES]
            inner_upper_lip_scores = scores[self.INNER_UPPER_LIP_INDICES]
            inner_lower_lip_keypoints = keypoints[self.INNER_LOWER_LIP_INDICES]
            inner_lower_lip_scores = scores[self.INNER_LOWER_LIP_INDICES]

            # 获取眼部对应关键点
            left_upper_eye_keypoints = keypoints[self.LEFT_UPPER_EYE_INDICES]
            left_upper_eye_scores = scores[self.LEFT_UPPER_EYE_INDICES]
            left_lower_eye_keypoints = keypoints[self.LEFT_LOWER_EYE_INDICES]
            left_lower_eye_scores = scores[self.LEFT_LOWER_EYE_INDICES]
            right_upper_eye_keypoints = keypoints[self.RIGHT_UPPER_EYE_INDICES]
            right_upper_eye_scores = scores[self.RIGHT_UPPER_EYE_INDICES]
            right_lower_eye_keypoints = keypoints[self.RIGHT_LOWER_EYE_INDICES]
            right_lower_eye_scores = scores[self.RIGHT_LOWER_EYE_INDICES]
            
            for keypoints_ in [
                'inner_upper_lip_keypoints', 
                'inner_lower_lip_keypoints', 
                'left_upper_eye_keypoints', 
                'left_lower_eye_keypoints', 
                'right_upper_eye_keypoints', 
                'right_lower_eye_keypoints'
            ]:
                color = [0, 0, 255] if 'upper' in keypoints_ else [255, 0, 0]
                scores_ = eval(f"{keypoints_.replace('_keypoints', '_scores')}")
                keypoints_ = eval(keypoints_)
                self._draw_landmarks(img, keypoints_, scores_, color, self.thre)
            
                mouth_width = keypoints[self.INNER_CORNER_LIP_INDICES[0]][0] - keypoints[self.INNER_CORNER_LIP_INDICES[1]][0]
                mouth_height = keypoints[self.INNER_UPPER_LIP_INDICES[1]][1] - keypoints[self.INNER_LOWER_LIP_INDICES[1]][1]
                mouth_open = (mouth_height / (mouth_width + 1e-5)) > self.yawn_thre
                scores_qualified = all([
                    scores[self.INNER_CORNER_LIP_INDICES[0]] > self.thre,
                    scores[self.INNER_CORNER_LIP_INDICES[1]] > self.thre,
                    scores[self.INNER_UPPER_LIP_INDICES[1]] > self.thre,
                    scores[self.INNER_LOWER_LIP_INDICES[1]] > self.thre,
                ])
                yawn = mouth_open and scores_qualified
                text = 'yawn' if yawn else ''
                cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                left_eye_width = keypoints[self.LEFT_CORNER_EYE_INDICES[0]][0] - keypoints[self.LEFT_CORNER_EYE_INDICES[1]][0]
                left_eye_height = keypoints[self.LEFT_UPPER_EYE_INDICES[1]][1] - keypoints[self.LEFT_LOWER_EYE_INDICES[1]][1]
                left_eye_closed = (left_eye_height / (left_eye_width + 1e-5)) < self.eye_closed_thre
                right_eye_width = keypoints[self.RIGHT_CORNER_EYE_INDICES[0]][0] - keypoints[self.RIGHT_CORNER_EYE_INDICES[1]][0]
                right_eye_height = keypoints[self.RIGHT_UPPER_EYE_INDICES[1]][1] - keypoints[self.RIGHT_LOWER_EYE_INDICES[1]][1]
                right_eye_closed = (right_eye_height / (right_eye_width + 1e-5)) < self.eye_closed_thre
                scores_qualified = all([
                    scores[self.LEFT_CORNER_EYE_INDICES[0]] > self.thre,
                    scores[self.LEFT_CORNER_EYE_INDICES[1]] > self.thre,
                    scores[self.LEFT_UPPER_EYE_INDICES[1]] > self.thre,
                    scores[self.LEFT_LOWER_EYE_INDICES[1]] > self.thre,
                    scores[self.RIGHT_CORNER_EYE_INDICES[0]] > self.thre,
                    scores[self.RIGHT_CORNER_EYE_INDICES[1]] > self.thre,
                    scores[self.RIGHT_UPPER_EYE_INDICES[1]] > self.thre,
                    scores[self.RIGHT_LOWER_EYE_INDICES[1]] > self.thre,
                ])
                eye_closed = left_eye_closed and right_eye_closed and scores_qualified
                text = 'eye closed' if eye_closed else ''
                cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            result_generator = self.inferencer(img, show=False)
            result = next(result_generator)
            keypoints = np.array(result['predictions'][0][0]['keypoints'])  
            scores = np.array(result['predictions'][0][0]['keypoint_scores'])
            img_name = None if is_video else imgs[img_indice].name
            self._save(img, img_name, keypoints, scores, save_dir)

        if is_video:
            self.writer.release()
            imgs.release()

        print(f'Results saved in {save_dir}')


if __name__ == '__main__':
    """
    基于mmpose API的关键点检测推理代码, mmpose内部实现的是目标检测+关键点检测
    同时这里实现了, 如果是人脸关键点检测的话, 可以判断是否打哈欠或者闭眼
    """
    inferencer = M_MMPoseInferencer()  # 环境conda activate openmmlab
    inferencer(inputs='inputs/videos/yawn-eyeclosed.mp4', save_dir='outputs/videos')
    