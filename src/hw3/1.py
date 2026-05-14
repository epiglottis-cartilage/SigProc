from typing import Any
import time
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage.measure import label, regionprops


timing_results = {}


def timed(name: str):
    """Decorator to time functions and store results."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            timing_results[name] = elapsed
            return result

        return wrapper

    return decorator


@timed("Image loading")
def load_image(path):
    """加载图像并转换为双精度灰度图"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img.astype(np.float64)


@timed("Median filtering")
def apply_median_filters(img):
    """Apply 3x3 and 5x5 median filters."""
    median_3x3 = ndimage.median_filter(img, size=3)
    median_5x5 = ndimage.median_filter(img, size=5)
    return median_3x3, median_5x5


@timed("Signal and noise profile plotting")
def plot_intensity_profiles(raw, median3, median5, center_x, center_y):
    """Task (1a): Plot 1D intensity profiles through a particle center."""
    # Extract horizontal line through the particle center
    line_raw = raw[center_y, :]
    line_m3 = median3[center_y, :]
    line_m5 = median5[center_y, :]

    x = np.arange(len(line_raw))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, line_raw, "k-", alpha=0.7, label="Raw")
    ax.plot(x, line_m3, "b-", label="3x3 Median")
    ax.plot(x, line_m5, "r-", label="5x5 Median")
    ax.axvline(
        center_x, color="gray", linestyle="--", alpha=0.5, label="Particle center"
    )
    ax.set_xlabel("Pixel Position")
    ax.set_ylabel("Intensity")
    ax.set_title(f"1D Intensity Profile at y={center_y} (through particle center)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hw3_part1_1a_intensity_profile.png", dpi=150)
    plt.close()

    return line_raw, line_m3, line_m5


@timed("Signal and noise profile plotting")
def plot_noise_characterization(raw, bg_y):
    """Task (1b): Plot background noise histogram and fit Gaussian model."""
    bg_line = raw[bg_y, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(
        bg_line, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black"
    )
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title(f"Background Noise Histogram at y={bg_y}")
    axes[0].grid(True, alpha=0.3)

    # Fit Gaussian
    mu = np.mean(bg_line)
    sigma = np.std(bg_line)
    x_fit = np.linspace(bg_line.min(), bg_line.max(), 200)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_fit - mu) / sigma) ** 2
    )
    axes[0].plot(
        x_fit,
        gaussian,
        "r-",
        linewidth=2,
        label=f"Gaussian fit: mu={mu:.2f}, sigma={sigma:.2f}",
    )
    axes[0].legend()

    # Q-Q plot or noise model comparison
    axes[1].hist(
        raw.ravel(),
        bins=256,
        range=(0, 256),
        density=True,
        alpha=0.6,
        color="gray",
        label="Full image",
    )
    axes[1].hist(
        bg_line,
        bins=50,
        density=True,
        alpha=0.8,
        color="steelblue",
        edgecolor="black",
        label=f"Background line (y={bg_y})",
    )
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_title("Noise Characterization")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hw3_part1_1b_noise_characterization.png", dpi=150)
    plt.close()

    print(f"Background noise: mu = {mu:.4f}, sigma = {sigma:.4f}")
    print(f"Signal-to-Noise Ratio (estimated): SNR ~ {mu / sigma:.2f}")

    return mu, sigma


@timed("Pseudo-color mapping (grayscale and size-based)")
def pseudocolor_grayscale(img, title, filename):
    """Task (2): Map grayscale image to pseudo-color."""
    # Normalize to [0, 1]
    img_norm = (img - img.min()) / (img.max() - img.min())

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img_norm, cmap="turbo")  # Continuous palette
    ax.set_title(title)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Intensity")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


@timed("Image segmentation")
def segment_otsu(median5):
    """Task (3a): Otsu thresholding to create binary mask."""
    # Convert to uint8 for Otsu
    img_uint8 = np.clip(median5, 0, 255).astype(np.uint8)

    # Otsu thresholding
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = (binary > 0).astype(np.uint8)

    # Show denoised and mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(median5, cmap="gray")
    axes[0].set_title("Denoised Image (5x5 Median)")
    axes[0].axis("off")

    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Binary Mask (Otsu)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("hw3_part1_3a_segmentation.png", dpi=150)
    plt.close()

    return binary_mask


@timed("Size measurement and histogram generation")
def analyze_particles(binary_mask, median5):
    """Task (3b & 3c): Measure particles and create size-based pseudo-color."""
    # Label connected components
    labeled = label(binary_mask)
    regions = regionprops(labeled)

    print(f"Number of particles detected: {len(regions)}")

    # Calculate equivalent diameters
    diameters = []
    for region in regions:
        area = region.area
        de = np.sqrt(area / np.pi)
        diameters.append(de)

    diameters = np.array(diameters)

    if len(diameters) == 0:
        print("No particles detected!")
        return None, None

    # Size-based pseudo-color mapping
    size_colored = np.zeros((*binary_mask.shape, 3))
    # Create color map based on diameter
    diam_min, diam_max = diameters.min(), diameters.max()

    for region in regions:
        area = region.area
        de = np.sqrt(area / np.pi)
        # Normalize diameter to [0, 1] for colormap
        if diam_max > diam_min:
            norm_size = (de - diam_min) / (diam_max - diam_min)
        else:
            norm_size = 0.5
        # Use turbo colormap
        color = plt.cm.turbo(norm_size)[:3]
        for coord in region.coords:
            size_colored[coord[0], coord[1]] = color

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(size_colored)
    ax.set_title("Size-Based Pseudo-Color Mapping")
    ax.axis("off")

    # Add colorbar for sizes
    sm = plt.cm.ScalarMappable(
        cmap="turbo", norm=plt.Normalize(vmin=diam_min, vmax=diam_max)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Equivalent Diameter (pixels)")

    plt.tight_layout()
    plt.savefig("hw3_part1_3b_size_pseudocolor.png", dpi=150)
    plt.close()

    # Task (3c): Size histogram
    mean_d = np.mean(diameters)
    std_d = np.std(diameters)

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = max(15, int(np.sqrt(len(diameters))))
    ax.hist(diameters, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(
        mean_d, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_d:.2f}"
    )
    ax.axvline(
        mean_d + std_d, color="orange", linestyle=":", linewidth=2, label=f"Mean + 1sigma"
    )
    ax.axvline(
        mean_d - std_d, color="orange", linestyle=":", linewidth=2, label=f"Mean - 1sigma"
    )
    ax.set_xlabel("Equivalent Diameter (pixels)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Particle Size Distribution (Mean = {mean_d:.2f} +/- {std_d:.2f} pixels)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hw3_part1_3c_size_histogram.png", dpi=150)
    plt.close()

    print(f"Particle size statistics: {mean_d:.2f} +/- {std_d:.2f} pixels")
    print(f"Size range: [{diameters.min():.2f}, {diameters.max():.2f}] pixels")

    return diameters, size_colored


def print_timing_table():
    """Task (4): Print timing table."""
    print("\n" + "=" * 60)
    print("TIMING TABLE (Unit: ms)")
    print("=" * 60)
    total = 0
    for task, elapsed in timing_results.items():
        print(f"{task:<45s} {elapsed:>10.2f}")
        total += elapsed
    print("-" * 60)
    print(f"{'Total pipeline execution':<45s} {total:>10.2f}")
    print("=" * 60)


def main():
    image_path = "./resource/HW3_img/particles_4k.tiff"

    print("=" * 60)
    print("HW3: Computational Imaging - Particle Analysis Pipeline")
    print("=" * 60)

    # Load image
    img = load_image(image_path)
    print(f"Image loaded: {img.shape}, dtype={img.dtype}")

    # Median filtering
    median3, median5 = apply_median_filters(img)
    print("Median filtering completed (3x3 and 5x5)")

    # Task (1a): Intensity profiles through a large particle
    # Using the suggested point (2080, 1228)
    center_x, center_y = 2080, 1228
    print(
        f"\nTask (1a): Intensity profile at y={center_y} (particle center x={center_x})"
    )
    plot_intensity_profiles(img, median3, median5, center_x, center_y)

    # Task (1b): Background noise characterization
    # Pick a background line - using y=500 as a background region
    bg_y = 1
    print(f"\nTask (1b): Background noise characterization at y={bg_y}")
    plot_noise_characterization(img, bg_y)

    # Task (2): Pseudo-color mapping
    print("\nTask (2): Pseudo-color mapping")
    pseudocolor_grayscale(img, "Raw Image - Pseudo-color", "hw3_part1_2a_raw_pseudocolor.png")
    pseudocolor_grayscale(
        median5,
        "Denoised Image (5×5 Median) - Pseudo-color",
        "hw3_part1_2b_denoised_pseudocolor.png",
    )

    # Task (3a): Segmentation
    print("\nTask (3a): Image segmentation with Otsu thresholding")
    binary_mask = segment_otsu(median5)

    # Task (3b & 3c): Particle analysis
    print("\nTask (3b & 3c): Particle size analysis")
    diameters, size_colored = analyze_particles(binary_mask, median5)

    # Task (4): Timing table
    print_timing_table()

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
