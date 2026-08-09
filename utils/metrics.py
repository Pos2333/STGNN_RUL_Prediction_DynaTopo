# ============================================================
# utils/metrics.py —— 评估指标
# ============================================================
# 提供模型评估用的指标:
#   1. RMSE（均方根误差）
#   2. NASA C-MAPSS 非对称评分（官方评估标准）
# ============================================================

import numpy as np
import torch


# ============================================================
# 1. RMSE（均方根误差）
# ============================================================
def compute_rmse(y_pred, y_true):
    """
    计算 RMSE（均方根误差）

    参数:
        y_pred: 预测值 (numpy 或 torch)
        y_true: 真实值 (numpy 或 torch)

    返回:
        rmse: 标量
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    return rmse


# ============================================================
# 2. NASA C-MAPSS 非对称评分（官方评估指标）
# ============================================================
def compute_nasa_score(y_pred, y_true):
    """
    计算 NASA C-MAPSS 官方非对称评分

    评分规则（与损失函数一致，但此处用于评估）:
      - d = pred - true
      - d < 0（过早预测）: exp(-d/13) - 1
      - d >= 0（过晚预测）: exp(d/10) - 1

    参数:
        y_pred: 预测值 (numpy 或 torch)
        y_true: 真实值 (numpy 或 torch)

    返回:
        total_score: 所有样本的评分总和（官方用总和而非均值）
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # 误差
    d = y_pred - y_true

    # 分段计算评分
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,   # 过早预测
        np.exp(d / 10.0) - 1.0      # 过晚预测
    )

    # 官方评估：返回总和
    return np.sum(scores)


# ============================================================
# 3. 综合评估函数（一次性输出 RMSE 和 Score）
# ============================================================
def evaluate_metrics(y_pred, y_true, print_result=True):
    """
    计算并输出 RMSE 和 NASA Score

    参数:
        y_pred: 预测值 (numpy 或 torch)
        y_true: 真实值 (numpy 或 torch)
        print_result: 是否打印结果

    返回:
        rmse, score: 两个标量
    """
    rmse = compute_rmse(y_pred, y_true)
    score = compute_nasa_score(y_pred, y_true)

    if print_result:
        print(f"  📊 RMSE: {rmse:.4f}")
        print(f"  📊 NASA Score: {score:.4f}")

    return rmse, score


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 评估指标自测")

    # 模拟数据：3个发动机的预测与真实 RUL
    pred = np.array([120.0, 50.0, 8.0])
    true = np.array([130.0, 45.0, 12.0])

    print(f"预测值: {pred}")
    print(f"真实值: {true}")

    rmse, score = evaluate_metrics(pred, true)
