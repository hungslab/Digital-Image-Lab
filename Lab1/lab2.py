# 2.选择两幅图像，一幅是物体图像，一幅是背景图像，采用正确的图像代数运算方法，分别实现图像叠加、混合图像的分离和图像的局部显示效果。
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取两幅图像
foreground = cv2.imread('images/1.jpg', 0)
background = cv2.imread('images/2.jpg', 0)

# 图像叠加
merged_image = cv2.add(foreground, background)

# 图像分离
sub_background_image = cv2.subtract(merged_image, foreground)

# 图像的局部显示
# 创建一个256x256的全零数组，将40到200行和40到200列的元素设置为1
B = np.zeros((500, 500))
B[40:201, 40:201] = 1

broken_view = background * B

# 显示图像
images = [foreground ,background , merged_image,sub_background_image ,broken_view]
title = ['Foreground', 'Background', 'Merged Image', 'Subtract Background Image Image', 'Broken view']

for i in range(0, len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(title[i])

plt.show()
