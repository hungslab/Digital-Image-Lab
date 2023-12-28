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
    image = cv2.imread('images/lena.png', 0)
    # 添加高斯噪声
    gaussian_noise_img = add_gussian_noise(image)
    # 添加椒盐噪声
    salt_pepper_noise = add_salt_pepper_noise(image)

    # 高斯滤波
    filtered_image_gaussian_gaussian = cv2.GaussianBlur(gaussian_noise_img,(3,3),1)
    # 使用中值滤波
    filtered_image_gaussian_median = cv2.medianBlur(gaussian_noise_img, 3)
    # 均值滤波
    filtered_image_gaussian_mean = cv2.blur(gaussian_noise_img, (3, 3))

    # 高斯滤波
    filtered_image_salt_pepper_guassian = cv2.GaussianBlur(salt_pepper_noise,(3,3),1)
    # 使用中值滤波
    filtered_image_salt_pepper_median = cv2.medianBlur(salt_pepper_noise, 3)
    # 均值滤波
    filtered_image_salt_pepper_mean = cv2.blur(salt_pepper_noise, (3, 3))

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    titles = ['加入高斯噪声后的图像', '高斯噪声高斯滤波后的结果', '高斯噪声中值滤波后的结果', '高斯噪声均值滤波后的结果', 
              '加入椒盐噪声后的图像', '椒盐噪声高斯滤波后的结果', '椒盐噪声中值滤波后的结果', '椒盐噪声均值滤波后的结果']
    images = [gaussian_noise_img, filtered_image_gaussian_gaussian, filtered_image_gaussian_median, 
              filtered_image_gaussian_mean, salt_pepper_noise, filtered_image_salt_pepper_guassian, filtered_image_salt_pepper_median, filtered_image_salt_pepper_mean]
    for i in range(len(titles)):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i], cmap='gray'), plt.title(titles[i]), plt.axis('off')
    plt.show()
