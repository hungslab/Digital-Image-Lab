# 3.	选择一幅图像，添加周期性噪声，选择合适的图像处理技术复原此降质图像。
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(img):  # 添加周期性噪声
    rows, cols = img.shape
    noise = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            noise[i, j] = img[i, j] + 40 * np.sin(40 * i) + 40 * np.sin(40 * j)
    return noise

def GBEF(img, W = 30, d0 = 50):  # 高斯带阻滤波器
    I_double = img.astype(float)  # 转换为 double 类型
    s = np.fft.fftshift(np.fft.fft2(I_double))  # 进行傅立叶变换和频谱平移
    a, b = s.shape  # 获取频域图像的大小
    a0 = a // 2  # 求图像的中心点坐标
    b0 = b // 2
    for i in range(a):
        for j in range(b):
            distance = np.sqrt((i - a0)**2 + (j - b0)**2)
            # 根据高斯带阻滤波器公式H(u,v)=1-e^-(1/2)[(D^2(u,v)-D^20)/D(u,v)*W]
            h = 1 - np.exp(-0.5 * ((distance**2 - d0**2) / (distance * W))**2)
            s[i, j] = h * s[i, j]  # 频域图像乘以滤波器的系数
    # s 中包含滤波后的频域图像,如果需要获取空域图像，可以进行逆傅立叶变换和逆平移
    filtered_image = np.fft.ifft2(np.fft.ifftshift(s)).real  # 获取空域图像
    return filtered_image

if __name__ == '__main__':
    # 读取图像并转换为灰度图像
    img = cv2.imread('images/lena.png', 0)
    # 添加周期性噪声
    noise = add_noise(img)
    # 将高斯带阻滤波器的带宽W设置为30 将高斯带阻滤波器的截止频率D0设置为50
    image_G_filtering = GBEF(noise, 30, 50)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(131), plt.title('原始图像'), plt.imshow(img, cmap='gray')
    plt.subplot(132), plt.title('带正弦噪声图像'), plt.imshow(noise, cmap='gray')
    plt.subplot(133), plt.imshow(image_G_filtering, 'gray'), plt.title('高斯带阻滤波图像')
    plt.show()