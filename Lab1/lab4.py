# 4.将一幅图像分别旋转45°和90°，与原图像对比，观察它们的区别。
import cv2
import matplotlib.pyplot as plt

# 读取图像
I = cv2.imread('1.jpg')

# 图像逆时针旋转45度
M = cv2.getRotationMatrix2D((I.shape[1] / 2, I.shape[0] / 2), 45, 1)
print(M)
J = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))

# 图像旋转90度出界的部分不被截出
M = cv2.getRotationMatrix2D((I.shape[1] / 2, I.shape[0] / 2), 90, 1)
print(M)
K = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]), borderMode=cv2.BORDER_REPLICATE)

images = [I, J, K]
title = ['Original Image', '45 Degree', '90 Degree']
for i in range(0, len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(title[i])

# 显示图像
plt.show()
