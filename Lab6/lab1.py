# 1.	选择一幅图像，分别利用Prewitt算子、Sobel算子及LOG算子显示图像的边缘。对比分析结果。

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 读取图像
img = cv2.imread('images/1.jpg', 0)
 
# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(img, cv2.CV_16S, kernelx)
y = cv2.filter2D(img, cv2.CV_16S, kernely)
# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Sobel算子
sx = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求一阶导
sy = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求一阶导
abssX = cv2.convertScaleAbs(sx)
abssY = cv2.convertScaleAbs(sy)
Sobel = cv2.addWeighted(abssX, 0.5, abssY, 0.5, 0)

# LOG算子

# 先通过高斯滤波降噪
gaussian = cv2.GaussianBlur(img, (3, 3), 0)
# 再通过拉普拉斯算子做边缘检测
dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
LOG = cv2.convertScaleAbs(dst)


# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
 
# 显示图形
titles = [u'原始图像', u'Prewitt算子', u'Sobel算子', u'LOG算子']
images = [img, Prewitt, Sobel, LOG]
for i in range(len(images)):
    plt.subplot(1, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
