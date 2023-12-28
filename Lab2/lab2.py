# 对图2.2 的两幅图像分别做旋转，观察原图的傅里叶频谱与旋转后的傅里叶频谱的对应 关系。
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 构造长方形原始图像
I = np.zeros((256, 256))
I[88:169, 124:132] = 1

# 显示原始图像
plt.subplot(2, 4, 1), plt.imshow(I, cmap='gray'), plt.title('Original Image')
# 求原始图像的傅里叶频谱
J = np.fft.fft2(I)
F = np.abs(J)
J1 = np.fft.fftshift(F)
plt.subplot(2, 4, 2), plt.imshow(J1, cmap='gray', vmin=0, vmax=50), plt.title('Original Image Spectrum')
# 对原始图像进行旋转
J = cv2.rotate(I, cv2.ROTATE_90_CLOCKWISE)
# 显示旋转后图像
plt.subplot(2, 4, 3), plt.imshow(J, cmap='gray'), plt.title('Rotated Image')
# 求旋转后图像的傅里叶频谱
J1 = np.fft.fft2(J)
F = np.abs(J1)
J2 = np.fft.fftshift(F)
plt.subplot(2, 4, 4), plt.imshow(J2, cmap='gray', vmin=0, vmax=50), plt.title('Rotated Image Spectrum')
# 构造正方形方形原始图像
I = np.zeros((256, 256))
I[50:206, 50:206] = 1
# 显示原始图像
plt.subplot(2, 4, 5), plt.imshow(I, cmap='gray'), plt.title('Original Image')
# 求原始图像的傅里叶频谱
J = np.fft.fft2(I)
F = np.abs(J)
J1 = np.fft.fftshift(F)
plt.subplot(2, 4, 6), plt.imshow(J1, cmap='gray', vmin=0, vmax=50), plt.title('Original Image Spectrum')
# 对原始图像进行旋转
height, width = I.shape
M  = cv2.getRotationMatrix2D((width/2,height/2), 45, 1.0)
dst = cv2.warpAffine(I, M, (width,height))
# 显示旋转后图像
plt.subplot(2, 4, 7), plt.imshow(dst, cmap='gray'), plt.title('Rotated Image')
# 求旋转后图像的傅里叶频谱
J1 = np.fft.fft2(dst)
F = np.abs(J1)
J2 = np.fft.fftshift(F)
plt.subplot(2, 4, 8), plt.imshow(J2, cmap='gray', vmin=0, vmax=50), plt.title('Rotated Image Spectrum')
plt.show()