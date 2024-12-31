import cv2
img = cv2.imread("OST_009.png",1)
print(img.shape)
img = cv2.resize(img, (240, 428))
print(img.shape)
cv2.imwrite("../workspace/OST_009_croped.jpg", img)