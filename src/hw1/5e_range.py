import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image_path = "./resource/HW1_img/Prob5_img.tif"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if original_img is None:
    raise FileNotFoundError(f"无法读取图像文件：{image_path}")


# ===================== 修复：分段灰度增强函数（向量化，无递归） =====================
def emphasize_region(img):
    # 创建空数组存储结果
    enhanced = np.zeros_like(img, dtype=np.float32)

    # 分段1：0~25 → 压缩（x/5）
    mask1 = img <= 25
    enhanced[mask1] = img[mask1] / 5 * 2

    # 分段2：25~100 → 拉伸（核心增强区间）
    mask2 = (img > 25) & (img <= 100)
    enhanced[mask2] = (img[mask2] - 25) * 2 + 10
    # 分段3：>100 → 压缩
    mask3 = img > 100
    enhanced[mask3] = (img[mask3] - 100) / 3 + 160

    # 转换为uint8图像类型（0-255）
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# 执行增强
enhanced_img = emphasize_region(original_img)

# ===================== 可视化 =====================
plt.figure(figsize=(9, 6))  # 调整尺寸更美观

# 子图1：增强后图像
plt.subplot(1, 2, 1)
plt.imshow(enhanced_img, cmap="gray")
plt.title("Range Emphasized Image")
plt.axis("off")

# 子图2：直方图（修复后无报错）
plt.subplot(1, 2, 2)
plt.hist(enhanced_img.ravel(), bins=256, range=[0, 255], color="gray")
plt.title("Histogram")
plt.xlabel("Gray Level (0-255)")
plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show()
