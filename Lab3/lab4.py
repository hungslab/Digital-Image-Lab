# 4.	选择一幅图像，使用频域高通滤波技术对图像滤波。
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
I = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
I = cv2.normalize(I.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Normalize to [0, 1]

m, n = I.shape
M, N = 2 * m, 2 * n
u = np.arange(-M//2, M//2)
v = np.arange(-N//2, N//2)
U, V = np.meshgrid(u, v) # 网格矩阵

# 频域中心的距离, 根据理想高通滤波器产生公式,当D(i,j)<=D0,置为0
D = np.sqrt(U**2 + V**2)
D0 = 40
H = np.array(D > D0, dtype=float) # 理想高通滤波

J = np.fft.fftshift(np.fft.fft2(I, s=(H.shape[0], H.shape[1])))
K = J * H
L = np.fft.ifft2(np.fft.ifftshift(K))
L = np.abs(L[:m, :n])  # Retain the original image size

plt.subplot(131), plt.imshow(np.log(1 + np.abs(J)), cmap='gray'), plt.title('Frequency Domain')
plt.subplot(132), plt.imshow(I, cmap='gray'), plt.title('Original Image')
plt.subplot(133), plt.imshow(L, cmap='gray'), plt.title('Filtered Image')
plt.show()
