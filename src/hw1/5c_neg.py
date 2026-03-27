import cv2
import matplotlib.pyplot as plt

image_path = "./resource/HW1_img/Prob5_img.tif"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_img is None:
    raise FileNotFoundError()

# -------------------------- 2. 生成灰度负片（核心：灰度反转） --------------------------
# 公式：负片像素 = 255 - 原始像素（自动适配uint8类型，无需额外转换）
negative_img = 255 - original_img

# -------------------------- 3. 显示原始图像+直方图 / 负片图像+直方图 --------------------------
plt.figure(figsize=(14, 8))

# 子图1：原始灰度图像
plt.subplot(2, 2, 1)
plt.imshow(original_img, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

# 子图2：原始图像直方图
plt.subplot(2, 2, 2)
plt.hist(original_img.ravel(), bins=256, range=[0, 255], color="gray")
plt.title("Histogram of Original Image")
plt.xlabel("Gray Level (0-255)")
plt.ylabel("Pixel Count")

# 子图3：灰度负片图像
plt.subplot(2, 2, 3)
plt.imshow(negative_img, cmap="gray")
plt.title("Negative Image (Gray-level Inversion)")
plt.axis("off")

# 子图4：负片图像直方图
plt.subplot(2, 2, 4)
plt.hist(negative_img.ravel(), bins=256, range=[0, 255], color="gray")
plt.title("Histogram of Negative Image")
plt.xlabel("Gray Level (0-255)")
plt.ylabel("Pixel Count")

# 调整布局并显示
plt.tight_layout()
plt.show()
