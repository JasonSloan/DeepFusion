import cv2
import numpy as np
import torch
from model import RRDBNet
import contextlib
import time
import pickle


class Profile(contextlib.ContextDecorator):
    """计时代码，可删"""
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class ESRGAN():
    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loadnet['params_ema'], strict=True)
        model.eval()
        model = model.to("cuda")
        self.model = model.half()
        self.model(torch.zeros(1, 3, 64, 64, dtype=torch.float16).to("cuda"))

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"input shape:{img.shape}")
        img = img.astype(np.float32)
        img = img / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to("cuda")
        img = img.half()
        return img


    def post_process(self, output_img):
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)
        return output


    def inference(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = self.preprocess(img)
        with dt:
            with torch.no_grad():
                output_img = self.model(img)
        output = self.post_process(output_img)
        return output


if __name__ == '__main__':
    imgs_paths = ['inputs/OST_009_croped.png' for _ in range(10)]
    img_path = imgs_paths[0]
    model_path = "weights/RealESRGAN_x4plus.pth"
    esrgan = ESRGAN(model_path)
    dt = Profile()
    seen = 0
    for img_path in imgs_paths:
        if img_path.endswith("png") or img_path.endswith("jpg"):
            seen += 1
            output = esrgan.inference(img_path)
            print(f"output shape:{output.shape}")
            # save_path = f"results/{seen}.jpg"
            # cv2.imwrite(save_path, output)
    t = dt.t * 1E3 / seen  # speeds per image
    print("Processed {} pitures, Average processing time: {:.2f} ms, FPS: {:.2f}".format(seen, t, 1 / (t / 1000)))
