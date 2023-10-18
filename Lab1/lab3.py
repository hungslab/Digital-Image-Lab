# 3.对一幅图像实现按比例缩小和不按比例任意缩小的效果，以及图像的成倍放大和不按比例放大效果。
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('1.jpg')

# 比例放大至1.5倍
B1 = cv2.resize(image, None, fx=1.5, fy=1.5)

# 非比例放大至420x384像素
B2 = cv2.resize(image, (420, 384))

# 比例放大至0.7倍
C1 = cv2.resize(image, None, fx=0.7, fy=0.7)

# 非比例放大至150x180像素
C2 = cv2.resize(image, (150, 180))

images = [image, B1, B2, C1, C2]
title= ['Original Image', '1.5x', '420x384', '0.7x', '150x180']
for i in range(0, len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(title[i])

# 显示图像
plt.show()
