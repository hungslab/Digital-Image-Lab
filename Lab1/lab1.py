# 1.灰度级别变换1.对一灰度图像，通过选择不同的灰度级变换函数s =T(r)实现图像的灰度范围线性扩展和非线性扩展，以及图像的灰度倒置和二值化。
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('images/1.jpg', 0)

# 线性扩展（线性拉伸）
def linear_stretch(image, a = 2.0, b = 0):
    normalized_image = np.float64(image)  
    linear_stretch = a * normalized_image + b
    return linear_stretch

# 非线性扩展（幂律变换）
def power_law(image, c = 1, gamma = 2):
    normalized_image = np.float64(image)  
    power_law = c * np.power(normalized_image, gamma)
    return power_law


# 灰度倒置
inverted_image = 255 - image

# 全局阈值法二值化
threshold_value = 127  # 阈值
_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# 全局阈值法二值化
threshold_value64 = 64  # 阈值
_, binary_image64 = cv2.threshold(image, threshold_value64, 255, cv2.THRESH_BINARY)

# 显示图像
images = [image ,linear_stretch(image) ,power_law(image) ,inverted_image , binary_image, binary_image64]
title = ['Original Image', 'Linear Stretch', 'Power Law', 'Inverted Image', '127 Binary Image', '64 Binary Image']

for i in range(0, len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(title[i])

plt.show()


