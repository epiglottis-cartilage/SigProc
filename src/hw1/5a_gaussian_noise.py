import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "./resource/HW1_img/Prob5_img.tif"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_img is None:
    raise FileNotFoundError()

def add_gaussian_noise(image, mean=0, variance=20):
    """
    为灰度图像添加高斯噪声
    :param image: 原始灰度图
    :param mean: 均值
    :param variance: 方差
    :return: 带噪声的灰度图
    """
    # 转换为浮点型，避免像素值溢出
    img_float = image.astype(np.float32)
    # 计算高斯噪声的标准差
    sigma = np.sqrt(variance)
    # 生成与图像同尺寸的高斯噪声
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    # 噪声与原图叠加
    noisy_img = img_float + gaussian_noise
    # 裁剪像素值到0-255（灰度图有效范围），并转回uint8类型
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


# 生成带高斯噪声的图像
noisy_img = add_gaussian_noise(original_img, mean=0, variance=64)

# -------------------------- 3. 显示图像与直方图 --------------------------
# 创建2x2子图布局
plt.figure(figsize=(14, 8))

# 子图1：原始灰度图像
plt.subplot(2, 2, 1)
plt.imshow(original_img, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")  # 隐藏坐标轴

# 子图2：原始图像灰度直方图
plt.subplot(2, 2, 2)
# ravel()将二维图像展平为一维数组，bins=256对应0-255所有灰度级
plt.hist(original_img.ravel(), bins=256, range=[0, 256], color="gray")
plt.title("Histogram of Original Image")
plt.xlabel("Gray Level (0-255)")
plt.ylabel("Pixel Count")

# 子图3：添加高斯噪声后的图像
plt.subplot(2, 2, 3)
plt.imshow(noisy_img, cmap="gray")
plt.title("Image with Gaussian Noise")
plt.axis("off")

# 子图4：噪声图像的灰度直方图
plt.subplot(2, 2, 4)
plt.hist(noisy_img.ravel(), bins=256, range=[0, 256], color="gray")
plt.title("Histogram of Noisy Image")
plt.xlabel("Gray Level (0-255)")
plt.ylabel("Pixel Count")

# 自动调整子图间距
plt.tight_layout()
# 显示所有图像和直方图
plt.show()
