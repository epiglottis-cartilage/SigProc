import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== Step 1: Prepare Noisy Image ======================
# Read grayscale image (replace with your image path)
img = cv2.imread("./resource/HW1_img/Prob6_img.tif", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found!")

# ✅ Simulate periodic noise (stripes/ripples) - skip this if your image already has noise
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
# Add sinusoidal periodic noise

# ====================== Step 2: 2D FFT + Frequency Shift ======================
# 1. Compute 2D Fast Fourier Transform
fft = np.fft.fft2(img)
# 2. Shift low frequencies to the CENTER (critical for visualization/filtering)
fft_shift = np.fft.fftshift(fft)
# 3. Compute magnitude spectrum (log scale for visualization)
magnitude = 20 * np.log(np.abs(fft_shift))

# ====================== Step 3: Locate Periodic Noise Spikes ======================
# Periodic noise = symmetric bright points in the spectrum
# We manually mark the noise coordinates (auto-detection is optional for beginners)
center_row, center_col = rows // 2, cols // 2
# Symmetric noise spike pairs (conjugate symmetry of FFT)
noise_points = [
    (center_col - 25, center_row - 25),
    (center_col + 25, center_row + 25),
]

# ====================== Step 4: Create Filter Mask (Remove Noise Spikes) ======================
# Gaussian mask (smooth filtering, no ringing artifacts)
mask = np.ones((rows, cols), dtype=np.float32)
radius = 3  # filter size
for c, r in noise_points:
    # Draw a Gaussian circle to suppress the noise spike
    cv2.circle(mask, (c, r), radius, 0, -1)
    cv2.line(mask, (0, r), (cols, r), 0, radius // 2)
    cv2.line(mask, (c, 0), (c, rows - 1), 0, radius // 2)

cv2.circle(mask, (center_col - 50, center_row - 50), radius, 0, -1)
cv2.circle(mask, (center_col + 50, center_row + 50), radius, 0, -1)
cv2.line(mask, (center_col, 0), (center_col, center_row - 5), 0, 1)
cv2.line(mask, (center_col, rows - 1), (center_col, center_row + 5), 0, 1)

# ====================== Step 5: Frequency Domain Filtering ======================
filtered_fft_shift = fft_shift * mask  # apply mask to the spectrum
filtered_magnitude = 20 * np.log(np.abs(filtered_fft_shift) + 1)  # avoid log(0)

# ====================== Step 6: Inverse FFT (Back to Spatial Domain) ======================
# 1. Inverse shift
filtered_fft = np.fft.ifftshift(filtered_fft_shift)
# 2. Inverse 2D FFT
denoised_img = np.fft.ifft2(filtered_fft)
# 3. Take real part + convert to 8-bit image
denoised_img = np.abs(denoised_img).astype(np.uint8)

# ====================== Visualization ======================
plt.figure(figsize=(16, 10))

# 1. Original noisy image
plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# 2. Frequency spectrum (noise spikes visible)
plt.subplot(2, 3, 2)
plt.imshow(magnitude, cmap="gray")
plt.title("Frequency Spectrum (Noise Spikes)")
plt.axis("off")

# 3. Filter mask
plt.subplot(2, 3, 3)
plt.imshow(mask, cmap="gray")
plt.title("Filter Mask (Black = Noise Removal)")
plt.axis("off")

# 4. Filtered spectrum
plt.subplot(2, 3, 4)
plt.imshow(filtered_magnitude, cmap="gray")
plt.title("Filtered Spectrum")
plt.axis("off")

# 5. Denoised image
plt.subplot(2, 3, 5)
plt.imshow(denoised_img, cmap="gray")
plt.title("Denoised Image (FFT Filtering)")
plt.axis("off")

plt.tight_layout()
plt.show()
