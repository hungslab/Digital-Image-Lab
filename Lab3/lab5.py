# 对第1题中的图像B做直方图规定化，并显示对比图像B与增强后的图像及对应直方图，规定直方图形状为图lena.png的图像直方图。
import numpy as np
import matplotlib.pyplot as plt
import cv2

#  定义函数，计算直方图累积概率
def histCalculate(src):
    row, col = np.shape(src)
    hist = np.zeros(256, dtype=np.float32)  # 直方图
    cumhist = np.zeros(256, dtype=np.float32)  # 累积直方图
    cumProbhist = np.zeros(256, dtype=np.float32)  # 累积概率probability直方图，即Y轴归一化
    for i in range(row):
        for j in range(col):
            hist[src[i][j]] += 1

    cumhist[0] = hist[0]
    for i in range(1, 256):
        cumhist[i] = cumhist[i-1] + hist[i]
    cumProbhist = cumhist/(row*col)
    return cumProbhist

# 定义函数，直方图规定化
def histSpecification(specImg, refeImg):  # specification image and reference image
    spechist = histCalculate(specImg)  # 计算待匹配直方图
    refehist = histCalculate(refeImg)  # 计算参考直方图
    corspdValue = np.zeros(256, dtype=np.uint8)  # correspond value
    # 直方图规定化
    for i in range(256):
        diff = np.abs(spechist[i] - refehist[i])
        matchValue = i
        for j in range(256):
            if np.abs(spechist[i] - refehist[j]) < diff:
                diff = np.abs(spechist[i] - refehist[j])
                matchValue = j
        corspdValue[i] = matchValue
    outputImg = cv2.LUT(specImg, corspdValue)
    return outputImg

if __name__ == '__main__':
    img = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
    # 读入参考图像
    img1 = cv2.imread('images/lena.png', cv2.IMREAD_GRAYSCALE)
    imgOutput = histSpecification(img, img1)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(3, 2, 1), plt.imshow(img, cmap='gray'), plt.title('原始图像B')
    plt.subplot(3, 2, 2), plt.hist(img.ravel(), 256, [0, 256]), plt.title('原始图像B的直方图')
    plt.subplot(3, 2, 3), plt.imshow(img1, cmap='gray'), plt.title('规定直方图像图像')
    plt.subplot(3, 2, 4), plt.hist(img1.ravel(), 256, [0, 256]), plt.title('规定直方图图像的直方图')
    plt.subplot(3, 2, 5), plt.imshow(imgOutput, cmap='gray'), plt.title('增强后的图像')
    plt.subplot(3, 2, 6), plt.hist(imgOutput.ravel(), 256, [0, 256]), plt.title('增强后图像的直方图')
    plt.show()



