# 对一幅图进行离散余弦变换，观察其余弦变换系数及余弦反变换后恢复图像。

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../sy1/1.jpg', 0)

# 进行离散余弦变换
dct = cv2.dct(np.float32(image))
# 进行余弦反变换
idct = cv2.idct(dct)

images = [image, np.log10(np.abs(dct) + 1), idct]
title = ['Original Image', 'DCT Coefficients', 'DCT Coefficients']

for i in range(0, len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(title[i])
plt.show()
