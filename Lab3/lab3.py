# 3. 选择一幅图像，分别使用prewitt算子、拉普拉斯算子进行滤波；
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('images/1.jpg')

# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

# Laplacian
dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst)

# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
 
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'Prewitt算子', u'Laplacian 算子']
images = [grayImage, Prewitt, Laplacian]
for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
