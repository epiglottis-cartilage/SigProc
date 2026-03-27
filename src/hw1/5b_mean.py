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
noisy_img = add_gaussian_noise(original_img, variance=64)

# -------------------------- 2. 均值滤波（3×3 / 5×5 / 9×9） --------------------------
# cv2.blur(图像, (窗口宽, 窗口高)) → 标准均值滤波（邻域平均）
filtered_3x3 = cv2.blur(noisy_img, (3, 3))  # 3×3窗口
filtered_5x5 = cv2.blur(noisy_img, (5, 5))  # 5×5窗口
filtered_9x9 = cv2.blur(noisy_img, (9, 9))  # 9×9窗口

# -------------------------- 3. 绘制：噪声图 + 3种滤波图 + 对应直方图 --------------------------
# 布局：4行2列（噪声/3×3/5×5/9×9，每一行是【图像+直方图】）
plt.figure(figsize=(16, 18))

# 行1：带噪声的图像 + 直方图
plt.subplot(4, 2, 1)
plt.imshow(noisy_img, cmap="gray")
plt.title("Noisy Image (Gaussian Noise)")
plt.axis("off")

plt.subplot(4, 2, 2)
plt.hist(noisy_img.ravel(), 256, [0, 256], color="black")
plt.title("Histogram - Noisy Image")
plt.xlabel("Gray Level")

# 行2：3×3均值滤波 + 直方图
plt.subplot(4, 2, 3)
plt.imshow(filtered_3x3, cmap="gray")
plt.title("Filtered Image - 3×3 Mean Filter")
plt.axis("off")

plt.subplot(4, 2, 4)
plt.hist(filtered_3x3.ravel(), 256, [0, 256], color="blue")
plt.title("Histogram - 3×3 Mean Filter")
plt.xlabel("Gray Level")

# 行3：5×5均值滤波 + 直方图
plt.subplot(4, 2, 5)
plt.imshow(filtered_5x5, cmap="gray")
plt.title("Filtered Image - 5×5 Mean Filter")
plt.axis("off")

plt.subplot(4, 2, 6)
plt.hist(filtered_5x5.ravel(), 256, [0, 256], color="orange")
plt.title("Histogram - 5×5 Mean Filter")
plt.xlabel("Gray Level")

# 行4：9×9均值滤波 + 直方图
plt.subplot(4, 2, 7)
plt.imshow(filtered_9x9, cmap="gray")
plt.title("Filtered Image - 9×9 Mean Filter")
plt.axis("off")

plt.subplot(4, 2, 8)
plt.hist(filtered_9x9.ravel(), 256, [0, 256], color="green")
plt.title("Histogram - 9×9 Mean Filter")
plt.xlabel("Gray Level")

plt.tight_layout()
plt.show()