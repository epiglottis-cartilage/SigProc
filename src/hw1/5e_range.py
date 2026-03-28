import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image_path = "./resource/HW1_img/Prob5_img.tif"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if original_img is None:
    raise FileNotFoundError(f"无法读取图像文件：{image_path}")


mask = (original_img > 25) & (original_img < 10)


def emphasize_region(x):
    if x <= 25:
        return x / 5
    elif 25 < x <= 100:
        return (x - 25) * 2 + emphasize_region(25)
    else:
        return (x - 100) / 5 + emphasize_region(100)


img = np.matrix(
    [[emphasize_region(x) for x in row] for row in original_img], dtype=np.uint8
)
# img = emphasize_region(original_img)

# 4. 可视化： 4行2列
plt.figure(figsize=(9, 16))

plt.imshow(img, cmap="gray")
plt.title("Range Emphasized Image")
plt.axis("off")

plt.tight_layout()
plt.show()
