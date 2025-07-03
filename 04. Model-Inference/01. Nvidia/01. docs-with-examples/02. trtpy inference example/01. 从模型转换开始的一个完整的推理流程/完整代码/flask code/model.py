import sr
import numpy as np
import cv2


import time
import contextlib
class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        return time.time()

class Model():
    def __init__(self, engine_path) -> None:
        self.inferInstance = sr.sr(engine=engine_path)

    def decode_binary(self, binaryImage):
        """将传入的二进制图片解码为ndarray"""
        image = np.frombuffer(binaryImage, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    
    def commit(self, binaryImage, id):
        dt = (Profile(), Profile(), Profile())
        # 1. 将base64编码的图像解码为ndarray
        imageArray = self.decode_binary(binaryImage)
        # 2. 调用推理引擎，必须显示传参
        result = self.inferInstance.forward(imageArray=imageArray, id=id)
        return result

if __name__ == "__main__":
    import cv2
    import numpy as np

    dt = Profile()

    imagePath = "/root/SuperResolution/flask/images/org.jpg"
    input_image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    with open(imagePath, "rb") as f:
        binaryImage = f.read()
    engine_path = "/root/SuperResolution/flask/engine.trtmodel"
    model = Model(engine_path=engine_path)
    if not model.inferInstance.valid:
        print("engine load failed")
    else:
        result = model.commit(binaryImage, "1")
       
        output_height, output_width = input_image.shape[0] * 4, input_image.shape[1] * 4
        with dt:
            recovered_image = result.output_array.reshape(output_height, output_width, 3)
            bytes_data = cv2.imencode(".png", recovered_image)[1].tobytes()

        with open("output_image.png", "wb") as f:
            f.write(bytes_data)
        print(f"commit time: {dt.t/10}")
        # cv2.imwrite("output_image.png", recovered_image)
