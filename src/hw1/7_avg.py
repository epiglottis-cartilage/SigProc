import cv2
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


rows, cols = imgs[0].shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

nums = [1, 2, 5, 10, 25, 50, 100]
# ====================== Visualization ======================
plt.figure(figsize=(16, 9))

for i, num in enumerate(nums):
    plt.subplot(2, 4, i + 1)
    plt.imshow(np.mean(imgs[:num], axis=0), cmap="gray")
    plt.title(f"average {num}")
    plt.axis("off")

plt.tight_layout()
plt.show()
