import cv2
from cv2 import data
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== Step 1: Prepare Noisy Image ======================
folder = "./resource/HW1_img/Prob7_img/"
imgs = [
    cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
    for f in os.listdir(folder)
    if f.endswith(".png")
]

imgs = [img[:30] for img in imgs]  # 前30row

rows, cols = imgs[0].shape


nums = [1, 2, 5, 10, 25, 50, 100]
# ====================== Visualization ======================
plt.figure(figsize=(16, 9))

for i, num in enumerate(nums):
    plt.subplot(2, 4, i + 1)
    img = np.mean(imgs[:num], axis=0)
    # 像素计数
    cont = np.bincount(img.astype(int).ravel(), minlength=256)
    # 正态拟合, 并且画出
    mean, std = cv2.meanStdDev(img)
    mean = mean[0, 0]
    std = std[0, 0]
    x = np.arange(0, 256)
    y = (
        img.size
        * (1 / (std * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x - mean) / std) ** 2)
    )
    plt.plot(x, y, color="red", label="Normal Fit")

    plt.hist(img.ravel(), bins=256, range=(0, 255), color="gray")
    plt.title(f"average {num}, {mean=:.2f}, {std=:.2f}")
    plt.xlabel("Gray Level (0-255)")
    plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show()
