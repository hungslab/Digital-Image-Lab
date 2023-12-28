# 2.	选择一幅图像，用不同的阈值选择方法分割图像，对比分析结果。

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_gray = cv2.imread('images/1.jpg', flags=0)  # 读取灰度图像
ret, th = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)  # 用cv2实现固定阈值分割

deltaT = 1  # 预定义值
histCV = cv2.calcHist([img_gray], [0], None, [256], [0, 256])  # 灰度直方图
grayScale = range(256)  # 灰度级 [0,255]
totalPixels = img_gray.shape[0] * img_gray.shape[1]  # 像素总数
totalGary = np.dot(histCV[:, 0], grayScale)  # 内积, 总和灰度值
T = round(totalGary / totalPixels)  # 平均灰度

while True:
    numG1, sumG1 = 0, 0
    for i in range(T):  # 计算 C1: (0,T) 平均灰度
        numG1 += histCV[i, 0]  # C1 像素数量
        sumG1 += histCV[i, 0] * i  # C1 灰度值总和
    numG2, sumG2 = (totalPixels - numG1), (totalGary - sumG1)  # C2 像素数量, 灰度值总和
    T1 = round(sumG1 / numG1)  # C1 平均灰度
    T2 = round(sumG2 / numG2)  # C2 平均灰度
    Tnew = round((T1 + T2) / 2)  # 计算新的阈值
    print("T={}, m1={}, m2={}, Tnew={}".format(T, T1, T2, Tnew))
    if abs(T - Tnew) < deltaT:  # 等价于 T==Tnew
        break
    else:
        T = Tnew

# 阈值处理
ret, imgBin = cv2.threshold(img_gray, T, 255, cv2.THRESH_BINARY)  # 阈值分割, thresh=T

ret2, imgOtsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)  # 阈值分割, thresh=T

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
images = [img_gray, th, imgBin, imgOtsu]
titles = ['image', "人工阈值分割法T=90", "迭代阈值分割 T={}".format(ret), "OTSU 阈值分割(T={})".format(round(ret2))]

for i in range(len(images)):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
