# ============================================================
# utils/loss_functions.py —— 损失函数集合
# ============================================================
# TODO 2 阶段仅包含:
#   1. MSE 损失（均方误差）
#   2. NASA C-MAPSS 非对称评分损失
#   3. 组合损失: 0.5 * MSE + 0.5 * NASA_Score
#
# ⚠️ 注意：LMMD 迁移学习损失在 TODO 5 才会加入，此处绝不提前写入！
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 1. NASA C-MAPSS 非对称评分函数
# ============================================================
def nasa_score_loss(y_pred, y_true):
    """
    计算 NASA C-MAPSS 数据集官方定义的非对称评分（用作损失函数）

    评分规则:
      - 如果预测 RUL < 真实 RUL（过早预测，d < 0）:
          score = exp(-d / 13) - 1
      - 如果预测 RUL >= 真实 RUL（过晚预测，d >= 0）:
          score = exp(d / 10) - 1

    含义：过晚预测（发动机已损坏才报警）比过早预测的惩罚更重
          （分母 10 < 13，所以 d>0 时指数增长更快）

    参数:
        y_pred: 模型预测的 RUL 值 [batch_size] 或 [batch_size, 1]
        y_true: 真实 RUL 值 [batch_size] 或 [batch_size, 1]

    返回:
        score_mean: 批次内平均评分（标量）
    """
    # 统一展平为一维
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # 误差 = 预测值 - 真实值
    d = y_pred - y_true

    # 分段计算非对称评分
    # torch.where(条件, 真分支, 假分支)
    scores = torch.where(
        d < 0,
        torch.exp(-d / 13.0) - 1.0,   # 过早预测
        torch.exp(d / 10.0) - 1.0      # 过晚预测
    )

    return scores.mean()


# ============================================================
# 2. MSE 损失（均方误差）
# ============================================================
class MSELoss(nn.Module):
    """
    MSE 损失封装，与 PyTorch 原生 nn.MSELoss 保持一致
    单独封装便于后续统一管理
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)


# ============================================================
# 3. 组合损失: 0.5 * MSE + 0.5 * NASA_Score
# ============================================================
class CombinedLoss(nn.Module):
    """
    TODO 2 阶段的主损失函数
    loss = MSE_WEIGHT * MSE + NASA_SCORE_WEIGHT * NASA_Score

    权重默认为 0.5 / 0.5，可在 configs/config.py 中调整
    """
    def __init__(self, mse_weight=0.5, nasa_weight=0.5):
        """
        参数:
            mse_weight: MSE 损失的权重
            nasa_weight: NASA Score 损失的权重
        """
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.nasa_weight = nasa_weight
        self.mse_fn = nn.MSELoss()

    def forward(self, y_pred, y_true):
        """
        计算组合损失

        参数:
            y_pred: 模型预测值 [batch_size, 1]
            y_true: 真实标签   [batch_size, 1]

        返回:
            total_loss: 组合损失（标量）
        """
        # MSE 部分
        mse = self.mse_fn(y_pred, y_true)

        # NASA Score 部分
        nasa = nasa_score_loss(y_pred, y_true)

        # 加权求和
        total_loss = self.mse_weight * mse + self.nasa_weight * nasa

        return total_loss


# ============================================================
# 4. LMMD 损失 —— 局部最大均值差异（TODO 5 迁移学习专用）
# ============================================================
# ⚠️ 严格遵守 TODO 约束：此函数仅在 TODO 5 阶段加入！


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算多带宽高斯核矩阵

    原理：使用多个不同带宽的高斯核能更好地捕捉不同尺度的分布差异。
    带宽以中位距离为基础，乘以 kernel_mul 的幂次进行缩放。

    参数:
        source:     源域特征 [N_s, D]
        target:     目标域特征 [N_t, D]
        kernel_mul: 带宽乘数因子
        kernel_num: 使用的核数量
        fix_sigma:  固定带宽（None 则自动计算）

    返回:
        kernels: [N_s+N_t, N_s+N_t] 核矩阵之和
    """
    n_s = int(source.size(0))
    n_t = int(target.size(0))
    total = torch.cat([source, target], dim=0)  # [N_s+N_t, D]

    # 计算成对 L2 距离平方矩阵
    L2_distance_square = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)

    # 自动计算带宽（使用中位距离的缩放）
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # 避免除零，确保分母安全
        n_total = n_s + n_t
        bandwidth = torch.sum(L2_distance_square.data) / (n_total ** 2 - n_total + 1e-8)
        bandwidth /= kernel_mul ** (kernel_num // 2)

    # 生成多个带宽的高斯核并求和
    kernel_val = []
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        kernel_val.append(torch.exp(-L2_distance_square / (bw + 1e-8)))

    return sum(kernel_val)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5):
    """
    最大均值差异 (Maximum Mean Discrepancy)

    计算公式: MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    其中 k 为高斯核函数。

    MMD 越大 → 两个分布差异越大。迁移学习的目标就是最小化 MMD。

    参数:
        source: 源域特征 [N_s, D]
        target: 目标域特征 [N_t, D]

    返回:
        mmd: 标量损失值
    """
    batch_size = int(source.size(0))
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num)

    # 拆分核矩阵为四个子块
    K_ss = kernels[:batch_size, :batch_size]          # 源-源
    K_tt = kernels[batch_size:, batch_size:]           # 目标-目标
    K_st = kernels[:batch_size, batch_size:]           # 源-目标
    K_ts = kernels[batch_size:, :batch_size]           # 目标-源

    loss = torch.mean(K_ss) + torch.mean(K_tt) \
        - torch.mean(K_st) - torch.mean(K_ts)
    return loss


def lmmd_loss(source_feat, target_feat, source_labels,
              num_subdomains=5, kernel_mul=2.0, kernel_num=5):
    """
    局部最大均值差异 (Local Maximum Mean Discrepancy)

    与全局 MMD 的区别：
      MMD 直接比较整个源域和目标域，忽略了 RUL 的内在结构。
      LMMD 先将 RUL 空间划分为多个子域（如早期、中期、晚期退化），
      在每个子域内分别对齐分布。这样做能更精细地匹配不同退化阶段的模式。

    步骤：
      1. 按源域标签的分布（分位数）将 RUL 范围划为 C 个子域
      2. 对每个子域，取出该范围内源域特征，与目标域特征计算 MMD
      3. 对各子域 MMD 取平均

    参数:
        source_feat:     源域特征 [B_s, D]（模型中间层输出）
        target_feat:     目标域特征 [B_t, D]
        source_labels:   源域 RUL 标签 [B_s, 1]（用于划分子域）
        num_subdomains:  子域数量（默认 5）
        kernel_mul:      高斯核带宽乘数
        kernel_num:      高斯核数量

    返回:
        mmd_avg: 所有子域 MMD 的平均值
    """
    source_labels_flat = source_labels.view(-1)

    # ---- 第1步：计算子域边界（按分位数等分） ----
    sorted_labels, _ = torch.sort(source_labels_flat)
    boundaries = []
    for i in range(1, num_subdomains):
        idx = int(len(sorted_labels) * i / num_subdomains)
        boundaries.append(sorted_labels[idx].item())

    # ---- 第2步：逐子域计算 MMD ----
    mmd_total = 0.0
    valid_count = 0

    for c in range(num_subdomains):
        # 确定当前子域在源域中的范围
        if c == 0:
            # 第一个子域: <= 第一个分界点
            mask_s = source_labels_flat <= boundaries[0]
        elif c == num_subdomains - 1:
            # 最后一个子域: > 最后一个分界点
            mask_s = source_labels_flat > boundaries[-1]
        else:
            # 中间子域: 在两个分界点之间
            mask_s = (source_labels_flat > boundaries[c - 1]) & \
                     (source_labels_flat <= boundaries[c])

        # 样本太少则跳过（避免数值不稳定）
        if mask_s.sum() < 3:
            continue

        feat_s_sub = source_feat[mask_s]  # 当前子域源域特征

        # 计算当前子域的 MMD
        mmd_val = mmd_loss(feat_s_sub, target_feat,
                           kernel_mul=kernel_mul, kernel_num=kernel_num)
        mmd_total += mmd_val
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=source_feat.device)

    return mmd_total / valid_count


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 损失函数自测")

    # 模拟预测值和真实值
    pred = torch.tensor([[100.0], [50.0], [10.0]])
    true = torch.tensor([[120.0], [40.0], [15.0]])

    print(f"预测值: {pred.flatten().tolist()}")
    print(f"真实值: {true.flatten().tolist()}")

    # 测试 MSE
    mse_loss = MSELoss()(pred, true)
    print(f"MSE 损失: {mse_loss.item():.4f}")

    # 测试 NASA Score
    nasa = nasa_score_loss(pred, true)
    print(f"NASA Score 损失: {nasa.item():.4f}")

    # 测试组合损失
    combined = CombinedLoss()(pred, true)
    print(f"组合损失 (0.5*MSE + 0.5*NASA): {combined.item():.4f}")
