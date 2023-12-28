# 1.	选择一幅图像，添加高斯噪声及椒盐噪声，观察对比使用均值滤波、中值滤波及修正后的阿尔法均值滤波的效果；
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('images/1.jpg', 0)
# 将图像转换为double类型
img = image.astype(float) / 255.0

#设置高斯分布的均值和方差
mean = 0
#设置高斯分布的标准差
sigma = 25
#根据均值和标准差生成符合高斯分布的噪声
gaussian_noise = np.random.normal(mean, sigma, img.shape)
gaussian_noise_img = np.clip(img + gaussian_noise, 0, 1)

# 均值滤波
filtered_image_mean = cv2.blur(gaussian_noise_img, (5, 5))

# 使用中值滤波
filtered_image_median = cv2.medianBlur((img * 255).astype(np.uint8), 3) / 255.0

titles = ['Original Image',  'Filtered Image - Mean Filter', 'Filtered Image - Median Filter']
images = [image, filtered_image_mean, filtered_image_median]
for i in range(len(titles)):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray'), plt.title(titles[i]), plt.axis('off')
plt.show()
