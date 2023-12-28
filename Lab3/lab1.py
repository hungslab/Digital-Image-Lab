# 1. 改变一幅图像A的灰度动态范围，使灰度范围在[0.2,0.8]的子区间内，得到图像B，通过直方图均衡化方法增强图像B，对比增强前后的图像；
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像A
image_A = cv2.imread('images/1.jpg', 0)
# 归一化图像A到[0, 1]的范围内
image_A_normalized = image_A / 255.0
# 限制灰度范围在[0.2, 0.8]的子区间内
image_A_constrained = np.clip(image_A_normalized, 0.2, 0.8)
# 将结果重新映射到[0, 255]的范围内
image_B = (image_A_constrained * 255).astype(np.uint8)
# 直方图均衡化
image_B_equalized = cv2.equalizeHist(image_B)


# 显示图像
images = [image_A, image_B,  image_B_equalized]
title = ['Original Image', 'Image B', 'Image B Equalized']
for i in range(0, len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(title[i])

plt.subplot(2,3,4), plt.hist(image_A.ravel(), 256, [0, 256])
plt.subplot(2,3,5), plt.hist(image_B.ravel(), 256, [0, 256])
plt.subplot(2,3,6), plt.hist(image_B_equalized.ravel(), 256, [0, 256])
plt.show()
