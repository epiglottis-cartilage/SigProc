import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    """加载图像并转换为双精度灰度图"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img.astype(np.float64)


def compress_by_magnitude(img, retain_pct):
    """
    基于幅度阈值保留指定百分比的傅里叶系数
    :param img: 输入图像 (float64)
    :param retain_pct: 保留系数百分比
    :return: 重建后的图像
    """
    # 计算二维FFT并移频
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    # 计算幅度并展平
    magnitudes = np.abs(F_shifted)
    flat_mag = magnitudes.ravel()
    total = flat_mag.size

    # 计算需要保留的系数个数 (至少保留1个)
    k = max(1, int(np.ceil(retain_pct / 100.0 * total)))

    # 找到第k大的幅度阈值
    threshold = np.partition(flat_mag, -k)[-k]

    # 保留幅度大于等于阈值的系数
    mask = magnitudes >= threshold
    F_compressed = F_shifted * mask

    # 逆变换并取实部
    F_ishift = np.fft.ifftshift(F_compressed)
    img_recon = np.real(np.fft.ifft2(F_ishift))

    # 裁剪到有效范围
    img_recon = np.clip(img_recon, 0, 255)
    return img_recon


def calculate_mae(original, reconstructed):
    """计算平均绝对误差 MAE"""
    return np.mean(np.abs(original - reconstructed))


def circular_lowpass_mask(shape, target_count):
    """
    创建圆形理想低通滤波器掩码，保留指定数量的系数
    :param shape: 图像尺寸 (H, W)
    :param target_count: 需要保留的系数个数
    :return: 布尔型掩码
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # 生成坐标网格并计算到中心的欧氏距离
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    flat_dist = distances.ravel()

    if target_count >= flat_dist.size:
        return np.ones(shape, dtype=bool)

    # 找到target_count个最近点的距离阈值
    r = np.partition(flat_dist, target_count - 1)[target_count - 1]
    mask = distances <= r

    # 处理边界上的重复距离，确保保留数量精确
    current_count = np.sum(mask)
    if current_count > target_count:
        boundary = distances == r
        boundary_idx = np.argwhere(boundary)
        remove = current_count - target_count
        to_remove = boundary_idx[:remove]
        mask[to_remove[:, 0], to_remove[:, 1]] = False

    return mask


def part1(img):
    """第一部分：生成2x2对比图"""
    percentages = [5, 1, 0.2]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 原始图像
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 重建图像
    for i, pct in enumerate(percentages):
        row = (i + 1) // 2
        col = (i + 1) % 2
        recon = compress_by_magnitude(img, pct)
        axes[row, col].imshow(recon, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'Reconstructed ({pct}% coefficients)')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def part2(img):
    """第二部分：MAE随保留比例变化曲线"""
    percentages = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    maes = []

    for pct in percentages:
        recon = compress_by_magnitude(img, pct)
        mae = calculate_mae(img, recon)
        maes.append(mae)
        print(f"Retention {pct}%: MAE = {mae:.4f}")

    plt.figure(figsize=(8, 6))
    plt.semilogx(percentages, maes, marker='o', linestyle='-', color='b')
    plt.xlabel('Retention Percentage (%)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE vs. Coefficient Retention Percentage')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # 添加对误差的描述
    # desc = (
    #     "As retention drops below 5%, MAE increases rapidly.\n"
    #     "The degradation is approximately exponential because\n"
    #     "image energy is concentrated in a few coefficients."
    # )
    # plt.text(0.15, 0.85, desc, transform=plt.gca().transAxes,
    #          fontsize=10, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def part3(img):
    """第三部分：幅度阈值 vs 低通滤波器对比"""
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)
    magnitudes = np.abs(F_shifted)
    total = magnitudes.size

    # 1% 幅度阈值
    target_count = int(np.ceil(1.0 / 100.0 * total))
    flat_mag = magnitudes.ravel()
    threshold = np.partition(flat_mag, -target_count)[-target_count]
    mask_mag = magnitudes >= threshold
    actual_count = np.sum(mask_mag)

    # 使用实际保留数量重建（幅度阈值）
    F_mag = F_shifted * mask_mag
    img_mag = np.real(np.fft.ifft2(np.fft.ifftshift(F_mag)))
    img_mag = np.clip(img_mag, 0, 255)

    # 相同数量的圆形低通滤波器
    mask_lp = circular_lowpass_mask(img.shape, int(actual_count))
    F_lp = F_shifted * mask_lp
    img_lp = np.real(np.fft.ifft2(np.fft.ifftshift(F_lp)))
    img_lp = np.clip(img_lp, 0, 255)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_mag, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'1% Magnitude Threshold ({actual_count} coeffs)')
    axes[0].axis('off')

    axes[1].imshow(img_lp, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'1% Low-Pass Filter ({actual_count} coeffs)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # 输出分析
    print("\nPart 3 Analysis:")
    print("The magnitude thresholding produces a sharper image.")
    print("Mathematical explanation:")
    print("- High-frequency components represent edges, textures, and fine details.")
    print("- Magnitude thresholding retains significant high-frequency coefficients,")
    print("  preserving important edges regardless of their spatial frequency.")
    print("- Low-pass filtering discards ALL high-frequency components beyond the cutoff,")
    print("  causing blurring because edge information (high freq) is removed uniformly.")


def main():
    image_path = "./resource/HW2_img/image-prob8.jpg"
    img = load_image(image_path)

    print("Running Part 1...")
    part1(img)

    print("Running Part 2...")
    part2(img)

    print("Running Part 3...")
    part3(img)


if __name__ == "__main__":
    main()
