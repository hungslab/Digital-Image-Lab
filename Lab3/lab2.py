# 2. 选择一幅图像，分别添加高斯噪声、椒盐噪声，观察对比使用不同大小（3*3及5*5）模板的均值滤波去噪效果；
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('images/1.jpg', 0)

# 添加高斯噪声：出现位置是一定的（每一点上），但噪声的幅值是随机的。
mean = 0
std_dev = 50
gaussian_noise = np.random.normal(mean, std_dev, image.shape)
image_with_gaussian_noise = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)

# 添加椒盐噪声：出现位置是随机的，但噪声的幅值是基本相同的。
salt_vs_pepper_ratio = 0.5
amount = 0.05
salt_vs_pepper_noise = np.random.rand(image.shape[0], image.shape[1])
image_with_salt_pepper_noise = image.copy()
image_with_salt_pepper_noise[salt_vs_pepper_noise < amount] = 0
image_with_salt_pepper_noise[salt_vs_pepper_noise > 1 - amount] = 255

# 均值滤波去噪
image_gaussian_denoised_3x3 = cv2.blur(image_with_gaussian_noise, (3, 3))
image_gaussian_denoised_5x5 = cv2.blur(image_with_gaussian_noise, (5, 5))
image_salt_pepper_denoised_3x3 = cv2.blur(image_with_salt_pepper_noise, (3, 3))
image_salt_pepper_denoised_5x5 = cv2.blur(image_with_salt_pepper_noise, (5, 5))

images = [image_with_gaussian_noise , image_gaussian_denoised_3x3, image_gaussian_denoised_5x5, 
          image_with_salt_pepper_noise, image_salt_pepper_denoised_3x3, image_salt_pepper_denoised_5x5]
title= ['image_with_gaussian_noise', 'Gaussian Denoised 3x3', 'Gaussian Denoised 5x5', 
        'image_with_salt_pepper_noise','Salt & Pepper Denoised 3x3', 'Salt & Pepper Denoised 5x5']
for i in range(0, len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(title[i])
plt.show()
