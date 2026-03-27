import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image_path = "./resource/HW1_img/Prob5_img.tif"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if original_img is None:
    raise FileNotFoundError(f"无法读取图像文件：{image_path}")


# 2. 定义伽马变换核心函数
def gamma_transformation(img, gamma):
    img_normalized = img / 255.0
    gamma_img = np.power(img_normalized, gamma)
    gamma_img = (gamma_img * 255).astype(np.uint8)
    return gamma_img


# 3. 设置3组伽马值
gamma_list = [0.4, 1.0, 2.5]
transformed_results = [gamma_transformation(original_img, g) for g in gamma_list]

# 4. 可视化： 4行2列
plt.figure(figsize=(20, 10))

# 循环绘制3组伽马变换结果（索引正常，无越界）
for idx, (gamma, img) in enumerate(zip(gamma_list, transformed_results)):
    # 变换后图像
    plt.subplot(4, 2, 2 * idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Transformed Image (Gamma = {gamma})")
    plt.axis("off")

    # 对应直方图
    plt.subplot(4, 2, 2 * idx + 2)
    plt.hist(img.ravel(), bins=256, range=[0, 255], color="gray")
    plt.title(f"Histogram (Gamma = {gamma})")
    plt.xlabel("Gray Level (0-255)")
    plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show()
