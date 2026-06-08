import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# ================== Sinusoidal Position Encoding ==================
def sinusoidal_position_encoding(seq_len, d_model):
    """
    生成 Sinusoidal Position Encoding 矩阵
    参数:
        seq_len: 序列长度
        d_model: 嵌入维度
    返回:
        pe: shape (seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ================== 二维向量旋转 ==================
def rotate_2d(x, theta):
    """
    对二维向量应用旋转矩阵
    参数:
        x: shape (..., 2)  最后一维为两个分量 (x0, x1)
        theta: 旋转角度 (弧度) shape (...,) 或标量
    返回:
        旋转后的向量，shape 同 x
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    x_rot = torch.stack([x[..., 0] * cos - x[..., 1] * sin,
                         x[..., 0] * sin + x[..., 1] * cos], dim=-1)
    return x_rot

# ================== 高维 RoPE ==================
def precompute_rope_freqs(d_model, seq_len, base=10000.0):
    """
    预计算 RoPE 中每对维度对应的频率 theta_i，以及每个位置的旋转角度
    参数:
        d_model: 嵌入维度（必须为偶数）
        seq_len: 序列最大长度
        base: 频率基数
    返回:
        freqs: shape (seq_len, d_model//2)  每个位置每对维度的旋转角度
    """
    assert d_model % 2 == 0, "d_model must be even for RoPE"
    # 计算每个维度对的频率指数
    i = torch.arange(0, d_model, 2, dtype=torch.float)
    theta = 1.0 / (base ** (i / d_model))  # (d_model//2,)
    # 每个位置的角度 = pos * theta
    positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
    freqs = positions * theta.unsqueeze(0)  # (seq_len, d_model//2)
    return freqs

def apply_rope(x, freqs):
    """
    对输入张量 x 应用 RoPE 旋转
    参数:
        x: shape (seq_len, batch_size, num_heads, d_per_head) 或 (seq_len, d_model)
        freqs: shape (seq_len, d_model//2) 每个位置每对维度的旋转角度
    返回:
        旋转后的 x，shape 相同
    """
    # 将最后一维按两两分组并 reshape 为 (..., d_model//2, 2)
    original_shape = x.shape
    # 确保最后一维是 d_model
    x_flat = x.view(-1, original_shape[-1])
    seq_len = x_flat.shape[0] if len(original_shape) == 2 else original_shape[0]
    d_model = original_shape[-1]
    # 分离成 (..., d_model//2, 2)
    x_reshape = x_flat.view(-1, d_model // 2, 2)  # (N, d_pair, 2)
    # 获取对应的旋转角度 (seq_len, d_pair)
    # 需要将 freqs 扩展到与 x_reshape 相同的 batch 维度
    if len(original_shape) == 3:  # (seq_len, batch, d_model)
        batch_size = original_shape[1]
        freqs_expanded = freqs.unsqueeze(1).expand(-1, batch_size, -1)  # (seq_len, batch, d_pair)
        freqs_expanded = freqs_expanded.reshape(-1, freqs_expanded.shape[-1])  # (seq_len*batch, d_pair)
    else:  # (seq_len, d_model)
        freqs_expanded = freqs.view(-1, freqs.shape[-1])  # (seq_len, d_pair)
    # 旋转
    x_rot = rotate_2d(x_reshape, freqs_expanded)  # (N, d_pair, 2)
    # 恢复原形状
    x_rot = x_rot.view(original_shape)
    return x_rot

# ================== 对比 E+pos 和 RoPE 的输入方式 ==================
def demo_input_methods(seq_len=4, d_model=8, vocab_size=20, batch_size=2):
    """
    演示两种位置注入方式：
        - E+pos: 词嵌入 + 位置编码
        - RoPE:  词嵌入（无位置编码），在 Q/K 上应用旋转
    返回两个输入表示供后续对比
    """
    # 模拟随机 token 索引
    token_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
    embedding = nn.Embedding(vocab_size, d_model)
    x_embed = embedding(token_ids)  # (seq_len, batch, d_model)

    # ---- E+pos 方式 ----
    pe = sinusoidal_position_encoding(seq_len, d_model)  # (seq_len, d_model)
    # 添加位置编码
    x_with_pos = x_embed + pe.unsqueeze(1)  # 广播到 batch 维度

    # ---- RoPE 方式 ----
    # 注意：RoPE 方式下，词嵌入本身不加位置编码，位置信息通过后续在 Q/K 上旋转注入
    # 这里仅返回原始词嵌入，旋转将在 attention 计算时进行
    x_rope = x_embed  # 未加位置编码

    print("=== E+pos 输入方式 ===")
    print("词嵌入 + 正弦位置编码（加法注入）")
    print(f"x_with_pos shape: {x_with_pos.shape}\n")

    print("=== RoPE 输入方式 ===")
    print("词嵌入（无位置编码），位置信息通过旋转注入到 Q 和 K 矩阵")
    print(f"x_rope shape: {x_rope.shape}\n")
    return x_with_pos, x_rope

# ================== 数值实验验证 RoPE 的相对位置性质 ==================
def experiment_relative_position():
    """
    构造两个 query 和 key，改变它们的绝对位置但保持相对位置不变，
    分别计算使用 E+pos 和 RoPE 时的 attention score（点积），
    验证 RoPE 的 score 只依赖于相对位置，而 E+pos 会依赖绝对位置。
    """
    d_model = 8
    seq_len = 6
    # 随机生成词嵌入矩阵 (seq_len, d_model) 并固定下来，使实验可复现
    torch.manual_seed(42)
    x_embed = torch.randn(seq_len, d_model)  # 模拟嵌入，没有位置信息

    # ----- E+pos 准备 -----
    pe = sinusoidal_position_encoding(seq_len, d_model)  # (seq_len, d_model)
    x_with_pos = x_embed + pe  # 加位置编码

    # ----- RoPE 准备 -----
    # 预计算所有位置的旋转角度
    freqs = precompute_rope_freqs(d_model, seq_len)
    # 对每个位置独立应用旋转（用于 Q 和 K）
    x_rotated = apply_rope(x_embed, freqs)  # (seq_len, d_model) 旋转后的表示

    # 定义两组位置对：(pos_q, pos_k) 和 (pos_q', pos_k')，使得相对位置相同
    # 相对位置 delta = pos_k - pos_q
    pos_pairs = [(0, 2), (1, 3), (2, 4)]
    delta = 2

    print("\n========== 数值实验：相对位置性质验证 ==========")
    print(f"固定相对位置 delta = {delta}")
    print("比较三组绝对位置下，attention score (点积) 的变化:\n")

    print("--- E+pos 方法 ---")
    scores_e_pos = []
    for q_pos, k_pos in pos_pairs:
        q = x_with_pos[q_pos]   # (d_model,)
        k = x_with_pos[k_pos]
        score = torch.dot(q, k).item()
        scores_e_pos.append(score)
        print(f"绝对位置 (query={q_pos}, key={k_pos})  score = {score:.4f}")
    print(f"Score 变化范围: max-min = {max(scores_e_pos)-min(scores_e_pos):.4f}\n")

    print("--- RoPE 方法 ---")
    scores_rope = []
    for q_pos, k_pos in pos_pairs:
        q_rot = x_rotated[q_pos]
        k_rot = x_rotated[k_pos]
        # 注意：RoPE 中 attention score 是旋转后的 Q 与 K 点积，由于旋转矩阵正交性，可推导出只依赖于相对位置
        score = torch.dot(q_rot, k_rot).item()
        scores_rope.append(score)
        print(f"绝对位置 (query={q_pos}, key={k_pos})  score = {score:.4f}")
    print(f"Score 变化范围: max-min = {max(scores_rope)-min(scores_rope):.4f}")
    print("理论上 RoPE 的 score 应完全相等（数值误差范围内）")

    # 更严格的验证：构造多个相对位置相同的点对并检查方差
    print("\n--- 批量验证 (相对位置固定为 2，取多组不同绝对位置) ---")
    test_pairs = [(0,2), (1,3), (2,4), (3,5)]
    scores_rope_batch = []
    for qp, kp in test_pairs:
        qr = x_rotated[qp]
        kr = x_rotated[kp]
        scores_rope_batch.append(torch.dot(qr, kr).item())
    print(f"RoPE 得分列表: {[f'{s:.6f}' for s in scores_rope_batch]}")
    std_val = torch.tensor(scores_rope_batch).std().item()
    print(f"标准差: {std_val:.8f} (接近 0 说明与绝对位置无关)")

    # 额外验证：相对位置不同时得分应当不同
    print("\n--- 对比不同相对位置 (RoPE) ---")
    pair_delta1 = (0, 2)  # delta=2
    pair_delta2 = (0, 3)  # delta=3
    score_d1 = torch.dot(x_rotated[0], x_rotated[2]).item()
    score_d2 = torch.dot(x_rotated[0], x_rotated[3]).item()
    print(f"相对位置 2 的得分: {score_d1:.4f}")
    print(f"相对位置 3 的得分: {score_d2:.4f}")
    print(f"差值: {score_d2 - score_d1:.4f} (应为非零，表示 RoPE 编码了不同的相对距离)")

# ================== 主程序 ==================
if __name__ == "__main__":
    # 1-3 的实现已在上方函数中给出
    # 4. 演示输入方式对比
    demo_input_methods()

    # 5. 数值实验验证相对位置性质
    experiment_relative_position()

    # 可选：可视化正弦位置编码
    pe_viz = sinusoidal_position_encoding(50, 128)
    plt.figure(figsize=(10, 6))
    plt.imshow(pe_viz.numpy(), aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.title("Sinusoidal Position Encoding (50 positions, 128 dims)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("sinusoidal_pe.png")
    print("\n已保存正弦位置编码热力图到 sinusoidal_pe.png")