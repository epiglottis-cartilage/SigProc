import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


# ============================================================
# Timing Infrastructure (with warmup + multiple runs averaging)
# ============================================================

timing_results = {}


def timed(name: str, runs: int = 5, warmup: int = 1):
    """Decorator to time functions with warmup and multiple-run averaging.

    Args:
        name: Task name for the timing table.
        runs: Number of timed runs to average over.
        warmup: Number of untimed warm-up runs (e.g. for JIT, cache).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Warm-up runs (not timed)
            for _ in range(warmup):
                func(*args, **kwargs)

            # Timed runs
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            mean_time = np.mean(times)
            std_time = np.std(times)
            timing_results[name] = {"mean": mean_time, "std": std_time, "times": times}
            return result

        return wrapper
    return decorator


def print_timing_table():
    """Print timing table with mean +/- std."""
    print("\n")
    print("TIMING TABLE (Unit: ms, averaged over multiple runs)")
    print("=" * 70)
    print(f"{'Task':<45s} {'Mean':<12s} {'Std':<12s}")
    print("-" * 70)
    total_mean = 0
    for task, data in timing_results.items():
        mean = data["mean"]
        std = data["std"]
        total_mean += mean
        print(f"{task:<45s} {mean:<12.2f} {std:<12.2f}")
    print(f"{'Total pipeline execution':<45s} {total_mean:<12.2f}")


# ============================================================
# Part 2 (1): DFT Matrix Construction and Visualization
# ============================================================

def dft_matrix(N):
    """Construct the N x N DFT matrix W_N where (W_N)_{k,n} = exp(-2*pi*i*k*n/N)."""
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W


@timed("DFT matrix construction", runs=1, warmup=0)
def visualize_dft_matrices():
    """Visualize DFT matrices for N = 8, 16, 32, 64."""
    sizes = [8, 16, 32, 64]
    fig, axes = plt.subplots(len(sizes), 4, figsize=(16, 16))

    for row, N in enumerate(sizes):
        W = dft_matrix(N)

        im0 = axes[row, 0].imshow(np.real(W), cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row, 0].set_title(f"N={N}: Real Part")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(np.imag(W), cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row, 1].set_title(f"N={N}: Imaginary Part")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(np.abs(W), cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].set_title(f"N={N}: Magnitude")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        im3 = axes[row, 3].imshow(np.angle(W), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        axes[row, 3].set_title(f"N={N}: Phase")
        axes[row, 3].axis("off")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

    plt.tight_layout()
    plt.savefig("hw3_part2_1_dft_matrices.png", dpi=150)
    plt.close()
    print("DFT matrix visualization saved to hw3_part2_1_dft_matrices.png")


# ============================================================
# Part 2 (2): 1D Signal DFT - Direct Matrix vs FFT
# ============================================================

def generate_signal(t):
    """Generate signal x(t) = sin(2*pi*50*t) + 0.5*sin(2*pi*120*t)."""
    return np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)


def compute_dft_direct(x):
    """Compute DFT using direct matrix multiplication."""
    N = len(x)
    W = dft_matrix(N)
    return W @ x


def compute_dft_fft(x):
    """Compute DFT using FFT."""
    return np.fft.fft(x)


def relative_error(x_matrix, x_fft):
    """Compute relative L2 error."""
    return np.linalg.norm(x_matrix - x_fft) / np.linalg.norm(x_fft)


def benchmark_1d_dft():
    """Benchmark 1D direct-matrix DFT vs FFT with averaging."""
    sizes = [32, 64, 128, 256, 512, 1024]
    results = []

    for N in sizes:
        t = np.linspace(0, 1, N, endpoint=False)
        x = generate_signal(t)

        # --- Direct matrix DFT (timed with averaging) ---
        # Warm-up
        for _ in range(1):
            compute_dft_direct(x)
        # Timed runs
        direct_times = []
        for _ in range(5):
            start = time.perf_counter()
            x_matrix = compute_dft_direct(x)
            direct_times.append((time.perf_counter() - start) * 1000)
        direct_mean = np.mean(direct_times)
        direct_std = np.std(direct_times)

        # --- FFT (timed with averaging) ---
        # Warm-up
        for _ in range(1):
            compute_dft_fft(x)
        # Timed runs
        fft_times = []
        for _ in range(5):
            start = time.perf_counter()
            x_fft = compute_dft_fft(x)
            fft_times.append((time.perf_counter() - start) * 1000)
        fft_mean = np.mean(fft_times)
        fft_std = np.std(fft_times)

        error = relative_error(x_matrix, x_fft)
        speedup = direct_mean / fft_mean if fft_mean > 0 else float("inf")

        results.append({
            "N": N,
            "direct_mean": direct_mean,
            "direct_std": direct_std,
            "fft_mean": fft_mean,
            "fft_std": fft_std,
            "speedup": speedup,
            "error": error,
        })

    # Print table
    print("\n" + "=" * 80)
    print("1D DFT Timing Comparison (5 runs, 1 warm-up)")
    print("=" * 80)
    print(f"{'N':<10} {'Direct (ms)':<20} {'FFT (ms)':<20} {'Speedup':<12} {'Rel. Error':<15}")
    print("-" * 80)
    for r in results:
        direct_str = f"{r['direct_mean']:.2f} +/- {r['direct_std']:.2f}"
        fft_str = f"{r['fft_mean']:.2f} +/- {r['fft_std']:.2f}"
        print(f"{r['N']:<10} {direct_str:<20} {fft_str:<20} {r['speedup']:<12.1f} {r['error']:<15.2e}")
    print("=" * 80)

    # Log-log plot
    N_vals = np.array([r["N"] for r in results])
    direct_means = np.array([r["direct_mean"] for r in results])
    fft_means = np.array([r["fft_mean"] for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(N_vals, direct_means,
                yerr=[r["direct_std"] for r in results],
                fmt="o-", capsize=4, label="Direct Matrix DFT", linewidth=2, markersize=8)
    ax.errorbar(N_vals, fft_means,
                yerr=[r["fft_std"] for r in results],
                fmt="s-", capsize=4, label="FFT", linewidth=2, markersize=8)

    # Fit O(N^2) to direct
    coeffs = np.polyfit(np.log(N_vals), np.log(direct_means), 1)
    fitted = np.exp(coeffs[1]) * N_vals ** coeffs[0]
    ax.loglog(N_vals, fitted, "--", label=f"Fit O(N^{coeffs[0]:.2f})", alpha=0.7)

    ax.set_xlabel("Signal Size N")
    ax.set_ylabel("Computation Time (ms)")
    ax.set_title("1D DFT: Direct Matrix vs FFT (Mean +/- Std)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("hw3_part2_2_timing_comparison.png", dpi=150)
    plt.close()
    print("Timing comparison saved to hw3_part2_2_timing_comparison.png")

    return results


# ============================================================
# Part 2 (3): Fourier Denoising of 1D Signal
# ============================================================

@timed("Fourier denoising", runs=1, warmup=0)
def fourier_denoising():
    """Demonstrate Fourier denoising on a 1D signal."""
    N = 1024
    t = np.linspace(0, 1, N, endpoint=False)

    # Clean signal
    x_clean = generate_signal(t)

    # Add Gaussian noise
    sigma_noise = 2.0
    np.random.seed(42)  # reproducible
    noise = np.random.normal(0, 1, N)
    x_noisy = x_clean + sigma_noise * noise

    # FFT of noisy signal
    X_noisy = np.fft.fft(x_noisy)
    power_spectrum = np.abs(X_noisy) ** 2

    # Threshold-based denoising
    threshold = 0.1 * np.max(power_spectrum)
    mask = power_spectrum > threshold
    X_filtered = X_noisy * mask

    # Inverse FFT
    x_filtered = np.real(np.fft.ifft(X_filtered))

    # Relative errors
    e_noisy = np.linalg.norm(x_noisy - x_clean) / np.linalg.norm(x_clean)
    e_filtered = np.linalg.norm(x_filtered - x_clean) / np.linalg.norm(x_clean)

    print(f"\nFourier Denoising Results:")
    print(f"  Relative error (noisy):    {e_noisy:.6f}")
    print(f"  Relative error (filtered): {e_filtered:.6f}")
    print(f"  Improvement factor:        {e_noisy / e_filtered:.2f}x")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, x_clean, "b-", label="Clean Signal", linewidth=1.5)
    axes[0, 0].plot(t, x_noisy, "r-", alpha=0.5, label="Noisy Signal", linewidth=0.5)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("Clean vs Noisy Signal")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    freqs = np.fft.fftfreq(N, d=1/N)
    axes[0, 1].stem(freqs[:N//2], power_spectrum[:N//2], linefmt="g-", markerfmt="go", basefmt=" ")
    axes[0, 1].axhline(threshold, color="r", linestyle="--", label=f"Threshold = {threshold:.2e}")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Power")
    axes[0, 1].set_title("Power Spectrum of Noisy Signal")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, x_clean, "b-", label="Clean Signal", linewidth=1.5)
    axes[1, 0].plot(t, x_noisy, "r-", alpha=0.3, label="Noisy Signal", linewidth=0.5)
    axes[1, 0].plot(t, x_filtered, "g-", label="Denoised Signal", linewidth=1.5)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_title(f"Fourier Denoising (Error: {e_filtered:.4f})")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].stem(freqs[:N//2], np.abs(X_filtered[:N//2])**2, linefmt="g-", markerfmt="go", basefmt=" ")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].set_title("Filtered Power Spectrum")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hw3_part2_3_fourier_denoising.png", dpi=150)
    plt.close()
    print("Fourier denoising results saved to hw3_part2_3_fourier_denoising.png")

    return e_noisy, e_filtered


# ============================================================
# Part 2 (4): 2D DFT - Direct Matrix vs FFT2
# ============================================================

def dft2_direct(image):
    """2D DFT via direct matrix multiplication: A_hat = W_M * A * W_N^T."""
    M, N = image.shape
    W_M = dft_matrix(M)
    W_N = dft_matrix(N)
    return W_M @ image @ W_N.T


def dft2_fft(image):
    """2D DFT via FFT2."""
    return np.fft.fft2(image)


def benchmark_2d_dft():
    """Benchmark 2D direct-matrix DFT vs FFT2 with averaging."""
    sizes = [64, 128, 256, 512, 1024]

    image_path = "./resource/HW3_img/particles_4k.tiff"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load {image_path}, using synthetic image")
        img = np.random.rand(1024, 1024)

    results = []

    for size in sizes:
        if img.shape[0] >= size and img.shape[1] >= size:
            image = img[:size, :size].astype(np.float64)
        else:
            image = cv2.resize(img, (size, size)).astype(np.float64)

        # --- Direct 2D DFT (with averaging) ---
        for _ in range(1):
            dft2_direct(image)
        direct_times = []
        for _ in range(5):
            start = time.perf_counter()
            A_matrix = dft2_direct(image)
            direct_times.append((time.perf_counter() - start) * 1000)
        direct_mean = np.mean(direct_times)
        direct_std = np.std(direct_times)

        # --- FFT2 (with averaging) ---
        for _ in range(1):
            dft2_fft(image)
        fft_times = []
        for _ in range(5):
            start = time.perf_counter()
            A_fft = dft2_fft(image)
            fft_times.append((time.perf_counter() - start) * 1000)
        fft_mean = np.mean(fft_times)
        fft_std = np.std(fft_times)

        error = np.linalg.norm(A_matrix - A_fft) / np.linalg.norm(A_fft)
        speedup = direct_mean / fft_mean if fft_mean > 0 else float("inf")

        results.append({
            "size": size,
            "direct_mean": direct_mean,
            "direct_std": direct_std,
            "fft_mean": fft_mean,
            "fft_std": fft_std,
            "speedup": speedup,
            "error": error,
        })

    # Print table
    print("\n")
    print("2D DFT Timing Comparison (5 runs, 1 warm-up)")
    print("=" * 80)
    print(f"{'Size':<10} {'Direct (ms)':<20} {'FFT2 (ms)':<20} {'Speedup':<12} {'Rel. Error':<15}")
    for r in results:
        direct_str = f"{r['direct_mean']:.2f} +/- {r['direct_std']:.2f}"
        fft_str = f"{r['fft_mean']:.2f} +/- {r['fft_std']:.2f}"
        print(f"{r['size']:<10} {direct_str:<20} {fft_str:<20} {r['speedup']:<12.1f} {r['error']:<15.2e}")

    # Log-log plot
    size_vals = np.array([r["size"] for r in results])
    direct_means = np.array([r["direct_mean"] for r in results])
    fft_means = np.array([r["fft_mean"] for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(size_vals, direct_means,
                yerr=[r["direct_std"] for r in results],
                fmt="o-", capsize=4, label="Direct Matrix 2D-DFT", linewidth=2, markersize=8)
    ax.errorbar(size_vals, fft_means,
                yerr=[r["fft_std"] for r in results],
                fmt="s-", capsize=4, label="FFT2", linewidth=2, markersize=8)

    coeffs = np.polyfit(np.log(size_vals), np.log(direct_means), 1)
    fitted = np.exp(coeffs[1]) * size_vals ** coeffs[0]
    ax.loglog(size_vals, fitted, "--", label=f"Fit O(N^{coeffs[0]:.2f})", alpha=0.7)

    ax.set_xlabel("Image Size (N x N)")
    ax.set_ylabel("Computation Time (ms)")
    ax.set_title("2D DFT: Direct Matrix vs FFT2 (Mean +/- Std)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("hw3_part2_4_2d_timing_comparison.png", dpi=150)
    plt.close()
    print("2D timing comparison saved to hw3_part2_4_2d_timing_comparison.png")

    return results


# ============================================================
# Main Function
# ============================================================

def main():
    print("=" * 80)
    print("HW3 Part 2: Discrete Fourier Transform Analysis")

    # Task 1: DFT Matrix Visualization
    print("\n--- Task 1: DFT Matrix Visualization ---")
    visualize_dft_matrices()

    # Task 2: 1D DFT Performance Comparison
    print("\n--- Task 2: 1D DFT Performance Comparison ---")
    benchmark_1d_dft()

    # Task 3: Fourier Denoising
    print("\n--- Task 3: Fourier Denoising ---")
    fourier_denoising()

    # Task 4: 2D DFT Performance Comparison
    print("\n--- Task 4: 2D DFT Performance Comparison ---")
    benchmark_2d_dft()

    # Print timing table (decorator-based)
    print_timing_table()

    print("\n" + "=" * 80)
    print("All Part 2 tasks completed successfully!")


if __name__ == "__main__":
    main()
