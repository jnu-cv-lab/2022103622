import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. 参数设置（图片已改为 1.jpg）
# --------------------------
BLOCK_SIZE = 32  # 图像分块大小（可调整为16/64）
ENERGY_PERCENT = 0.95  # 目标能量百分比（95%）
IMAGE_PATH = '1.jpg'  # 你的图片路径，已改为1.jpg

# --------------------------
# 2. 空域梯度法计算 frms
# --------------------------
def gradient_frms(block):
    block = block.astype(np.float32)
    # Sobel算子计算梯度
    grad_x = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 计算E[|∇I|²]和Var(I)
    e_grad_sq = np.mean(grad_mag**2)
    var_I = np.var(block)
    
    if var_I < 1e-6:  # 避免除以0
        return 0.0
    frms_sq = e_grad_sq / (4 * np.pi**2 * var_I)
    return np.sqrt(frms_sq)

# --------------------------
# 3. FFT法计算包含95%能量的最高频率
# --------------------------
def fft_95_percent_freq(block):
    block = block.astype(np.float32)
    # 二维FFT并移频
    fft_shift = np.fft.fftshift(np.fft.fft2(block))
    power_spectrum = np.abs(fft_shift)**2
    total_energy = np.sum(power_spectrum)
    
    if total_energy < 1e-6:
        return 0.0
    
    # 计算每个频率的距离（从中心）
    h, w = block.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h//2, w//2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 按频率从高到低累加能量
    unique_dists = np.sort(np.unique(dist))[::-1]
    cumulative_energy = 0.0
    cutoff_freq = 0.0
    for d in unique_dists:
        mask = (dist == d)
        cumulative_energy += np.sum(power_spectrum[mask])
        if cumulative_energy / total_energy >= ENERGY_PERCENT:
            cutoff_freq = d
            break
    
    # 归一化频率（相对于奈奎斯特频率）
    normalized_freq = cutoff_freq / (BLOCK_SIZE / 2)
    return normalized_freq

# --------------------------
# 4. 主流程：分块处理图像
# --------------------------
def main():
    # 读取灰度图像
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 无法读取图像 1.jpg，请把图片放在代码同一文件夹！")
        return
    h, w = img.shape
    print(f"✅ 成功读取图像 1.jpg，大小: {w}x{h}")

    # 分块数量
    num_blocks_h = h // BLOCK_SIZE
    num_blocks_w = w // BLOCK_SIZE
    print(f"📦 分块数量: {num_blocks_h}x{num_blocks_w}")

    grad_freqs = []
    fft_freqs = []

    # 遍历所有块
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y1, y2 = i*BLOCK_SIZE, (i+1)*BLOCK_SIZE
            x1, x2 = j*BLOCK_SIZE, (j+1)*BLOCK_SIZE
            block = img[y1:y2, x1:x2]

            if np.var(block) < 1e-6:  # 跳过纯色块
                continue

            # 计算两种方法的频率
            grad_freqs.append(gradient_frms(block))
            fft_freqs.append(fft_95_percent_freq(block))

    grad_freqs = np.array(grad_freqs)
    fft_freqs = np.array(fft_freqs)
    errors = fft_freqs - grad_freqs

    # --------------------------
    # 5. 结果可视化
    # --------------------------
    plt.figure(figsize=(12, 6))

    # 散点图对比
    plt.subplot(1, 2, 1)
    plt.scatter(grad_freqs, fft_freqs, alpha=0.5, s=10)
    plt.xlabel('梯度法 frms')
    plt.ylabel('FFT法 95%能量频率')
    plt.title('两种方法频率对比')
    # 理想一致线
    min_val, max_val = min(grad_freqs.min(), fft_freqs.min()), max(grad_freqs.max(), fft_freqs.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想一致线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 误差分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
    plt.xlabel('误差 (FFT频率 - 梯度频率)')
    plt.ylabel('块数量')
    plt.title('误差分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frequency_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 统计分析
    print(f"\n📊 统计结果:")
    print(f"梯度法平均频率: {np.mean(grad_freqs):.4f} ± {np.std(grad_freqs):.4f}")
    print(f"FFT法平均频率: {np.mean(fft_freqs):.4f} ± {np.std(fft_freqs):.4f}")
    print(f"平均绝对误差: {np.mean(np.abs(errors)):.4f}")
    print(f"相关系数: {np.corrcoef(grad_freqs, fft_freqs)[0,1]:.4f}")

if __name__ == "__main__":
    main()