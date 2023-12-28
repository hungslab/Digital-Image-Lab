import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gussian_noise(image):
    # 设置高斯分布的均值和方差
    mean = 0
    # 设置高斯分布的标准差
    sigma = 25
    # 根据均值和标准差生成符合高斯分布的噪声
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    gaussian_noise_img = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)
    return gaussian_noise_img

def add_uniform_noise(image):
    noiseUniform = np.random.uniform(-50, 50, image.shape)
    imgUniformNoise = image + noiseUniform
    imgUniformNoise = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgUniformNoise

def add_salt_pepper_noise(image):
    #设置添加椒盐噪声的数目比例
    s_vs_p = 0.5
    #设置添加噪声图像像素的数目
    amount = 0.05
    noisy_img = np.copy(image)
    #添加salt噪声
    num_salt = np.ceil(amount * image.size * s_vs_p)
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0],coords[1]] = 255
    #添加pepper噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0],coords[1]] = 0
    return noisy_img

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('images/1.jpg', 0)  # 以灰度模式读取图像
    # 添加均匀分布噪声
    noisy_image = add_uniform_noise(image) # 噪声的振幅
    # 添加高斯噪声
    gaussian_noise_img = add_gussian_noise(image)
    # 添加椒盐噪声
    salt_pepper_noise = add_salt_pepper_noise(image)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示原始图像和添加噪声后的图像
    images = [image, noisy_image, gaussian_noise_img, salt_pepper_noise]
    titles = ['原始图像', '添加均匀分布噪声后的图像', '添加高斯噪声后的图像', '添加椒盐噪声后的图像']
    for i in range(len(titles)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], cmap='gray'), plt.title(titles[i]), plt.axis('off')
    plt.show()
