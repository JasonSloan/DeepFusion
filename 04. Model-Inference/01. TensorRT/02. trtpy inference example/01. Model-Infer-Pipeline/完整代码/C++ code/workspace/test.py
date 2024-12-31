import sr
import cv2
import numpy as np


imagePath = "OST_009_croped.jpg"
imageArray = cv2.imread(imagePath, cv2.IMREAD_COLOR)
# print(input_image.shape)
output_height = imageArray.shape[0] * 4
output_width = imageArray.shape[1] * 4

engine = "engine.trtmodel"
# 由于是自定义的库，必须显示传参，必须使用engine=的方式
infer = sr.sr(engine=engine)
print(infer.valid)
# 由于是自定义的库，必须显示传参，必须使用image=的方式
for i in range(100):
    result = infer.forward(imageArray=imageArray, id="1")
# print(result.code)
output_image = np.array(result.output_vector, dtype=np.uint8).reshape(output_height, output_width, 3)
cv2.imwrite("output_image.png", output_image)


# output_image_vector = infer.forward(image=input_image)
# output_height = input_image.shape[0] * 4
# output_width = input_image.shape[1] * 4
# recovered_image = np.array(output_image_vector, dtype=np.uint8).reshape(output_height, output_width, 3)
# cv2.imwrite("output_image.png", recovered_image)

