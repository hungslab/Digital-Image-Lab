% 2.选择一幅图像，生成运动模糊并加入噪声的降质图像，观察对比逆滤波与维纳滤波的复原效果；
A = imread('object_image2_gray.png');

D = 0.1 * randn(size(A));  % 正态分布的随机噪声
E = imadd(uint8(C), uint8(D));  % 对降质图像附加噪声
F = deconvwnr(E, B);  % 不带参数的维纳滤波（逆滤波）复原
G = sum(D(:).^2) / sum(double(A(:)).^2);  % 计算噪声信噪比
H = deconvwnr(E, B, G);  % 带噪声信噪比参数的维纳滤波复原
I = abs(fftn(D)).^2;
J = real(ifftn(I));  % 计算噪声的自相关函数
K = abs(fftn(double(A))).^2;
L = real(ifftn(K));  % 计算信号的自相关函数
M = deconvwnr(E, B, J, L);  % 带自相关函数的维纳滤波复原

% 显示图像
figure;
subplot(2, 3, 1); imshow(A); title('原始图像');
subplot(2, 3, 2); imshow(E); title('降质图像');
subplot(2, 3, 3); imshow(F); title('不带参数的维纳滤波（逆滤波）复原');
subplot(2, 3, 4); imshow(H); title('带噪信比参数的维纳滤波复原');
subplot(2, 3, 5); imshow(M); title('带自相关函数参数的维纳滤波复原');