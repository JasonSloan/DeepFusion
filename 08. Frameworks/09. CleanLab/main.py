# 参考链接: https://docs.cleanlab.ai/stable/tutorials/object_detection.html
# pip install cleanlab
import os
import os.path as osp
import sys
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import cv2
import torch
from cleanlab.object_detection.filter import find_label_issues
from cleanlab.object_detection.summary import visualize
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")


class LabelIssueInspector:
    
    def __init__(self, root_dir, save_dir, keep_clss, weight, device):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.model = YOLO(weight).to(torch.device(device))
        self.all_classes = keep_clss
        self.class_names = self.model.names
        self._remove_old_create_new()
    
    def _remove_old_create_new(self):
        if osp.exists(self.save_dir):
            print(self.colorstr('yellow', f'Your save dir {self.save_dir} already exists, do you want to remove it? [y/n]'))
            if input().lower() == 'y':
                shutil.rmtree(self.save_dir)
            else:
                sys.exit()
        os.makedirs(self.save_dir, exist_ok=True)
        self.orig_save_dir = osp.join(self.save_dir, 'original')
        self.clean_save_dir = osp.join(self.save_dir, 'clean')
        os.makedirs(self.orig_save_dir, exist_ok=True)
        os.makedirs(self.clean_save_dir, exist_ok=True)
    
    def run(self):
        self._format()
        self._find_issues()
    
    def _format(self):
        formated_labels = []
        formated_predictions = []
        img_names = os.listdir(self.root_dir)
        img_names = [img_name for img_name in img_names if img_name.endswith('.jpg') or img_name.endswith('.png')]
        bar = tqdm(img_names, desc='Formatting labels and predictions')
        for img_name in bar: 
            img_path = osp.join(self.root_dir, img_name)
            self._format_labels(img_path, formated_labels)
            self._format_predictions(img_path, formated_predictions)
                            
        self.formated_labels = formated_labels
        self.formated_predictions = formated_predictions

    def _format_labels(self, img_path, formated_labels):
        img = cv2.imread(img_path)
        label_path = img_path.replace('images', 'labels').replace('jpg', 'txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()
        image_h, image_w, _ = img.shape
        bboxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0), dtype=np.int32)
        for line in lines:
            cls, xc, yc, w, h = map(float, line.strip().split())
            if cls not in self.all_classes:
                continue
            ltrb = self._xywh2ltrb([xc, yc, w, h], to_abs=True, image_w=image_w, image_h=image_h)
            if bboxes.size > 0:
                bboxes = np.concatenate([bboxes, np.array([ltrb])], axis=0)
                labels = np.concatenate([labels, np.array([cls])], axis=0)
            else:    
                bboxes = np.array([ltrb])
                labels = np.array([cls])
        formated_labels.append({
            'image': img_path,
            'bboxes': bboxes.astype(np.int32),
            'labels': labels.astype(np.int32)
        })
        
    def _format_predictions(self, img_path: str, formated_predictions: list) -> None:
        pred = self.model.predict(
            source=img_path,
            conf=0.4,
            iou=0.4,
            save=False,
            stream=False,
            verbose=False
        )[0]
        bboxes = [np.zeros((0, 5), dtype=np.float32) for _ in range(len(self.all_classes))]
        for box in pred.boxes:
            cls = box.cls.int().cpu().numpy().tolist()[0]
            confidence = box.conf.cpu().numpy().tolist()[0]
            x, y, w, h = box.xywh.cpu().numpy().tolist()[0]
            ltrb = self._xywh2ltrb([x, y, w, h], to_abs=False)
            box = np.array([[*ltrb, confidence]], dtype=np.float32)  # Shape (1, 5)
            if bboxes[cls].size > 0:
                bboxes[cls] = np.vstack([bboxes[cls], box])
            else:
                bboxes[cls] = box
        formated_predictions.append(bboxes)
    
    def _xywh2ltrb(self, xywh, to_abs=True, image_w=None, image_h=None):
        xc, yc, w, h = xywh
        left = xc - (w / 2)
        top = yc - (h / 2)
        right = xc + (w / 2)
        bottom = yc + (h / 2)
        if not to_abs:
            return [left, top, right, bottom]
        return [int(left * image_w), int(top * image_h), int(right * image_w), int(bottom * image_h)]  
    
    def _find_issues(self):
        print(self.colorstr('yellow', 'Finding label issues'))
        label_issue_idxes = find_label_issues(
            labels=self.formated_labels,
            predictions=self.formated_predictions,
            return_indices_ranked_by_score=True
        )
        print(self.colorstr('green', f'Found {len(label_issue_idxes)} label issues, saving them to {self.save_dir}'))
        for issue_idx in label_issue_idxes:
            image_name = osp.basename(self.formated_labels[issue_idx]['image'])
            orig_image_save_dir= osp.join(self.orig_save_dir, 'images')
            shutil.copy2(self.formated_labels[issue_idx]['image'], orig_image_save_dir)
            shutil.copy2(self.formated_labels[issue_idx]['image'].replace('images', 'labels').replace('jpg', 'txt'), \
                orig_image_save_dir.replace('images', 'labels'))
            
            clean_save_path= osp.join(self.clean_save_dir, image_name)
            visualize(
                self.formated_labels[issue_idx]['image'], 
                label=self.formated_labels[issue_idx], 
                prediction=self.formated_predictions[issue_idx], 
                class_names=self.class_names, 
                overlay=False,
                figsize=(28, 20),
                save_path=clean_save_path
            )
            print(self.colorstr('green', 'Done!'))
            
    def colorstr(self, *input):
        *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
        colors = {'black': '\033[30m',  # basic colors
                'red': '\033[31m',
                'green': '\033[32m',
                'yellow': '\033[33m',
                'blue': '\033[34m',
                'magenta': '\033[35m',
                'cyan': '\033[36m',
                'white': '\033[37m',
                'bright_black': '\033[90m',  # bright colors
                'bright_red': '\033[91m',
                'bright_green': '\033[92m',
                'bright_yellow': '\033[93m',
                'bright_blue': '\033[94m',
                'bright_magenta': '\033[95m',
                'bright_cyan': '\033[96m',
                'bright_white': '\033[97m',
                'end': '\033[0m',  # misc
                'bold': '\033[1m',
                'underline': '\033[4m'}
        return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
    
    @classmethod
    def parse_arg(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--root-dir', type=str, default='/your/images/root/dir', help='src images root dir')
        parser.add_argument('--save-dir', type=str, default='your/save/dir', help='save dir')
        parser.add_argument('--keep-clss', type=list, default=[0], help='specify the target classes that need to be kept')
        parser.add_argument('--weight', type=str, default='your/model/weight/path', help='weight path')
        parser.add_argument('--device', type=int, default=0, help='device number')
        opt = parser.parse_args()
        return opt


if __name__ == '__main__':
    inspector = LabelIssueInspector(**vars(LabelIssueInspector.parse_arg()))
    inspector.run()