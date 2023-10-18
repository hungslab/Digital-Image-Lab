# 对图 2.1（a）进行平移，观察原图的傅里叶频谱与平移后的傅里叶频谱的对应关系。
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("2.png", 0)  # 读取彩色图像(BGR)
image = np.float64(img)

rows, cols = image.shape

dx, dy = 100, 100  # dx=100 向右偏移量, dy=50 向下偏移量
MATx = np.float64([[1, 0, dx], [0, 1, 0]])  # 构造平移变换矩阵
MATy = np.float64([[1, 0, 0], [0, 1, dy]])

dstx = cv2.warpAffine(image, MATx, (cols, rows))  # 默认为黑色填充
dsty = cv2.warpAffine(image, MATy, (cols, rows))

# 傅里叶变换
images = [image, dstx, dsty]
title = ['Original Image', 'dx=100', 'dy=100']

# 显示原图
for i in range(0, len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(title[i])

# 傅里叶变换后的图像
for i in range(0, len(images)):
    f = np.fft.fft2(images[i])
    fshift = np.fft.fftshift(f)
    # fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))
    plt.subplot(2, 3, i + 1 + 3), plt.imshow(fimg, cmap='gray'), plt.title(title[i])

plt.show()
