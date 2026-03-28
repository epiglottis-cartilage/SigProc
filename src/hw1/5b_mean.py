import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "./resource/HW1_img/Prob5_img.tif"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_img is None:
    raise FileNotFoundError()


# -------------------------- 2. 均值滤波（3×3 / 5×5 / 9×9） --------------------------
# cv2.blur(图像, (窗口宽, 窗口高)) → 标准均值滤波（邻域平均）
filtered_3x3 = cv2.blur(original_img, (3, 3))  # 3×3窗口
filtered_5x5 = cv2.blur(original_img, (5, 5))  # 5×5窗口
filtered_9x9 = cv2.blur(original_img, (9, 9))  # 9×9窗口

# -------------------------- 3. 绘制：噪声图 + 3种滤波图 + 对应直方图 --------------------------
# 布局：4行2列（噪声/3×3/5×5/9×9，每一行是【图像+直方图】）
plt.figure(figsize=(16, 18))

# 行1：带噪声的图像 + 直方图
plt.subplot(4, 2, 1)
plt.imshow(original_img, cmap="gray")
plt.title("Noisy Image (Gaussian Noise)")
plt.axis("off")

plt.subplot(4, 2, 2)
plt.hist(original_img.ravel(), 256, [0, 256], color="black")
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
