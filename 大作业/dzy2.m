% 选择一幅图像，生成运动模糊并加入噪声的降质图像，观察对比逆滤波与维纳滤波的复原效果；
A = imread('lena.png');
A = rgb2gray(A);    %RGB图像转灰度图像
B = fspecial('motion', 40, 30);% 运动位移是40像素，角度是30°
C = imfilter(A, B, 'conv', 'circular');  % 进行运动模糊
D = 0.1 * randn(size(A));  % 正态分布的随机噪声
E = imadd(uint8(C), uint8(D));  % 对降质图像附加噪声
F = deconvwnr(E, B);  % 逆滤波复原
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


% 选择一幅图像，生成运动模糊并加入噪声的降质图像，观察对比逆滤波与维纳滤波的复原效果；

% 读取图像
originalImage = imread('lena.png');
grayImage = rgb2gray(originalImage);  % 将RGB图像转换为灰度图像

% 创建运动模糊核
motionBlurKernel = fspecial('motion', 40, 30); % 运动位移为40像素，角度为30°

% 进行运动模糊
blurredImage = imfilter(grayImage, motionBlurKernel, 'conv', 'circular');

% 添加高斯白噪声
noise = 0.1 * randn(size(grayImage)); % 正态分布的随机噪声
degradedImage = imadd(uint8(blurredImage), uint8(noise)); % 降质图像附加噪声

% 使用不同滤波方法进行图像复原
inverseFiltered = deconvwnr(degradedImage, motionBlurKernel); % 逆滤波复原

% 计算噪声信噪比
noiseSNR = sum(noise(:).^2) / sum(double(grayImage(:)).^2);

% 使用带噪声信噪比参数的维纳滤波进行复原
wienerFilteredWithSNR = deconvwnr(degradedImage, motionBlurKernel, noiseSNR);

% 计算噪声和信号的自相关函数
noiseAutoCorrelation = abs(fftn(noise)).^2; % 噪声的自相关函数
J = real(ifftn(noiseAutoCorrelation))

signalPower = abs(fftn(double(grayImage))).^2; % 信号的自相关函数
L = real(ifftn(signalPower));  % 计算信号的自相关函数

% 使用带自相关函数参数的维纳滤波进行复原
wienerFilteredWithCorrelation = deconvwnr(degradedImage, motionBlurKernel, J, L);


% 显示图像
figure;
subplot(2, 3, 1); imshow(grayImage); title('原始图像');
subplot(2, 3, 2); imshow(degradedImage); title('降质图像');
subplot(2, 3, 3); imshow(inverseFiltered); title('不带参数的维纳滤波（逆滤波）复原');
subplot(2, 3, 4); imshow(wienerFilteredWithSNR); title('带噪声信噪比参数的维纳滤波复原');
subplot(2, 3, 5); imshow(wienerFilteredWithCorrelation); title('带自相关函数参数的维纳滤波复原');




















