# 选择一幅图像，利用如下代码给图像添加噪声，设计陷波滤波器复原图像。
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(img):  # 添加周期性噪声
    sizec = img.shape
    w = 0.4 * 2 * np.pi  # 噪声的数字频率
    img_noise = img + 20 * np.ones((sizec[0], 1)) * np.sin(w * np.arange(1, sizec[1] + 1))
    return img_noise

def butterworthNRFilter(shape, radius=9, uk=60, vk=80, n=2):  # 巴特沃斯陷波带阻滤波器
    M, N = shape[1], shape[0]
    u, v = np.meshgrid(np.arange(M).all(), np.arange(N).all())
    Dm = np.sqrt((u - M // 2 - uk) ** 2 + (v - N // 2 - vk) ** 2)
    Dp = np.sqrt((u - M // 2 + uk) ** 2 + (v - N // 2 + vk) ** 2)
    D0 = radius
    n2 = n * 2
    kernel = (1 / (1 + (D0 / (Dm + 1e-6)) ** n2)) * (1 / (1 + (D0 / (Dp + 1e-6)) ** n2))
    return kernel

def imgFrequencyFilter(img, radius=10):
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    # 边缘填充
    imgPad = np.pad(img, ((0, img.shape[0]), (0, img.shape[1])), mode="reflect")
    # 中心化: f(x,y) * (-1)^(x+y)
    mask = np.ones(imgPad.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    # 在频率域修改傅里叶变换: 傅里叶变换 点乘 滤波器传递函数
    ifft = np.fft.ifft2(np.fft.fft2(imgPad * mask) * butterworthNRFilter(imgPad.shape, radius=9, uk=60, vk=80, n=2))  # 巴特沃斯陷波带阻滤波器
    M, N = img.shape[:2]
    mask2 = np.ones(imgPad.shape)
    mask2[1::2, ::2] = -1
    mask2[::2, 1::2] = -1
    ifftCenPad = ifft.real * mask2
     # 截取左上角，大小和输入图像相等
    imgFilter = ifftCenPad[:M, :N]
    imgFilter = np.clip(imgFilter, 0, imgFilter.max())
    imgFilter = np.uint8(normalize(imgFilter) * 255)
    return imgFilter

if __name__ == '__main__':
    # 读取图像并转换为灰度图
    img = cv2.imread('images/lena.png', 0)
    img_noise = add_noise(img)
    # 图像巴特沃斯陷波带阻滤波
    imgBNRF = imgFrequencyFilter(img_noise, radius=9)
    # 显示结果
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.axis('off'), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(img_noise, cmap='gray'), plt.axis('off'), plt.title('Degraded Image')
    plt.subplot(1, 3, 3), plt.title("ButterworthNR filter"), plt.axis('off'), plt.imshow(imgBNRF, cmap='gray')
    plt.show()
