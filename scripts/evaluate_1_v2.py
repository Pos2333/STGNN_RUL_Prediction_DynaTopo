# ============================================================
# scripts/evaluate_1_v2.py —— 单工况预测性能对比实验（v2：无 Transformer）
# ============================================================
# TODO 4: 在 FD001 测试集上公平对比 LSTM 与 STGNN (v2: MSTCN + GAT) 的预测性能
# 基于消融实验结果，STGNN 采用无 Transformer 变体
#
# 对比指标:
#   - RMSE（均方根误差，越低越好）
#   - NASA Score（C-MAPSS 官方非对称评分，越低越好）
#
# 用法:
#   python scripts/evaluate_1.py
# ============================================================

import os
import sys
import json
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    NUM_FEATURES, BATCH_SIZE, RANDOM_SEED,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from core_models.base_models import BasicLSTM
from core_models.stgnn_full import STGNN
from utils.metrics import compute_rmse, compute_nasa_score, evaluate_metrics

# ============================================================
# 固定随机种子
# ============================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1. 加载测试数据 + 图结构
# ============================================================
def load_test_data(subset='FD001', processed_dir='data/processed'):
    """
    加载预处理好的测试数据和图结构

    返回:
        test_loader: 测试 DataLoader
        edge_index:  图边索引（STGNN 需要）
        y_true_all:  全部真实标签（numpy，用于最后统一评估）
    """
    test_path = os.path.join(processed_dir, f'{subset}_test.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试数据: {test_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"找不到图结构: {graph_path}")

    # 加载数据
    test_data = np.load(test_path)
    X_test = test_data['X']  # [n_samples, W, N]
    y_test = test_data['y']  # [n_samples]

    # 加载图结构
    graph = torch.load(graph_path)
    edge_index = graph['edge_index']  # [2, num_edges]

    print(f"\n📂 测试数据加载完成 - {subset}")
    print(f"  样本数: {len(X_test)}, 特征形状: {X_test.shape[1:]}")
    print(f"  图边数: {edge_index.shape[1]}")
    print(f"  标签范围: [{y_test.min():.0f}, {y_test.max():.0f}]")

    # 转为 tensor
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # DataLoader（不打乱，保证预测顺序）
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=False)

    return test_loader, edge_index, y_test


# ============================================================
# 2. 加载并评估 LSTM 模型
# ============================================================
def evaluate_lstm(test_loader, device, model_path='saved_models/lstm_best_FD001.pt'):
    """
    加载 LSTM 最佳模型并在测试集上预测

    返回:
        rmse, score, y_pred_all, y_true_all, num_params
    """
    print(f"\n{'='*60}")
    print(f"  🔍 评估 BasicLSTM")
    print(f"{'='*60}")

    # 实例化模型
    model = BasicLSTM(
        input_dim=NUM_FEATURES,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    ).to(device)

    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 LSTM 模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  已加载权重 (Epoch {checkpoint['epoch']+1}, "
          f"Val Loss: {checkpoint['best_loss']:.4f})")

    num_params = sum(p.numel() for p in model.parameters())

    # 预测
    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            y_pred_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_batch.numpy())

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    rmse = compute_rmse(y_pred_all, y_true_all)
    score = compute_nasa_score(y_pred_all, y_true_all)

    print(f"  📊 RMSE: {rmse:.2f}")
    print(f"  📊 NASA Score: {score:.2f}")

    return rmse, score, y_pred_all, y_true_all, num_params


# ============================================================
# 3. 加载并评估 STGNN 模型
# ============================================================
def evaluate_stgnn(test_loader, edge_index, device,
                   model_path='saved_models/stgnn_v2_best_FD001.pt'):
    """
    加载 STGNN v2 最佳模型并在测试集上预测

    注意: STGNN 推理时需要传入 edge_index，且要按 batch 扩展
    """
    print(f"\n{'='*60}")
    print(f"  🔍 评估 STGNN v2 (MSTCN + GAT, 无 Transformer)")
    print(f"{'='*60}")

    # 实例化模型
    model = STGNN(
        num_sensors=14, num_op_settings=3,
        mstcn_channels=MSTCN_NUM_CHANNELS,
        mstcn_kernels=MSTCN_KERNEL_SIZES,
        mstcn_dropout=MSTCN_DROPOUT,
        gat_hidden=GAT_HIDDEN_DIM,
        gat_heads=GAT_HEADS,
        gat_dropout=GAT_DROPOUT,
        trans_d_model=TRANSFORMER_D_MODEL,
        trans_nhead=TRANSFORMER_NHEAD,
        trans_num_layers=TRANSFORMER_NUM_LAYERS,
        trans_dropout=TRANSFORMER_DROPOUT,
        use_transformer=False,
        fc_hidden=FC_HIDDEN_DIM
    ).to(device)

    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 STGNN 模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  已加载权重 (Epoch {checkpoint['epoch']+1}, "
          f"Val Loss: {checkpoint['best_loss']:.4f})")

    num_params = sum(p.numel() for p in model.parameters())

    # 预测
    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            # 注意: STGNN.forward() 内部会自动调用 repeat_edge_index_for_batch
            # 所以这里只需传入原始 edge_index（14 节点的图结构）
            edge_index_device = edge_index.to(device)

            y_pred = model(X_batch, edge_index_device)
            y_pred_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_batch.numpy())

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    rmse = compute_rmse(y_pred_all, y_true_all)
    score = compute_nasa_score(y_pred_all, y_true_all)

    print(f"  📊 RMSE: {rmse:.2f}")
    print(f"  📊 NASA Score: {score:.2f}")

    return rmse, score, y_pred_all, y_true_all, num_params


# ============================================================
# 4. 结果汇总与输出
# ============================================================
def print_comparison(lstm_results, stgnn_results):
    """
    打印 LSTM vs STGNN 的对比表格
    """
    lstm_rmse, lstm_score, _, _, lstm_params = lstm_results
    stgnn_rmse, stgnn_score, _, _, stgnn_params = stgnn_results

    # 计算提升幅度
    rmse_improve = (lstm_rmse - stgnn_rmse) / lstm_rmse * 100
    score_improve = (lstm_score - stgnn_score) / lstm_score * 100

    print(f"\n{'='*65}")
    print(f"  📊 FD001 单工况预测性能对比")
    print(f"{'='*65}")
    print(f"  {'模型':<30} {'RMSE ↓':>10} {'NASA Score ↓':>15} {'参数量':>10}")
    print(f"  {'-'*65}")
    print(f"  {'BasicLSTM':<30} {lstm_rmse:>10.2f} {lstm_score:>15.2f} {lstm_params:>10,}")
    print(f"  {'STGNN':<30} {stgnn_rmse:>10.2f} {stgnn_score:>15.2f} {stgnn_params:>10,}")
    print(f"  {'-'*65}")

    if rmse_improve > 0:
        print(f"  ✅ STGNN 的 RMSE 比 LSTM 降低了 {rmse_improve:.1f}%")
    else:
        print(f"  ⚠️ STGNN 的 RMSE 比 LSTM 升高了 {-rmse_improve:.1f}%")

    if score_improve > 0:
        print(f"  ✅ STGNN 的 NASA Score 比 LSTM 降低了 {score_improve:.1f}%")
    else:
        print(f"  ⚠️ STGNN 的 NASA Score 比 LSTM 升高了 {-score_improve:.1f}%")

    print(f"{'='*65}")

    return {
        'lstm': {'rmse': float(lstm_rmse), 'score': float(lstm_score), 'params': lstm_params},
        'stgnn': {'rmse': float(stgnn_rmse), 'score': float(stgnn_score), 'params': stgnn_params},
        'rmse_improvement_pct': float(rmse_improve),
        'score_improvement_pct': float(score_improve),
    }


# ============================================================
# 5. 保存评估结果
# ============================================================
def save_results(results, subset='FD001'):
    """保存评估结果到 logs/ 目录"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/evaluate_{subset}_{timestamp}.json'

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📝 评估结果已保存 → {log_path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print("=" * 65)
    print(f"  🧪 TODO 4: FD001 单工况预测性能对比实验 (v2: 无 Transformer)")
    print(f"  BasicLSTM vs STGNN (MSTCN + GAT)")
    print("=" * 65)

    # ---- 设备选择 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  评估设备: {device}")

    # ---- 1. 加载测试数据（两个模型共用） ----
    test_loader, edge_index, y_test = load_test_data(subset='FD001')

    # ---- 2. 评估 LSTM ----
    lstm_results = evaluate_lstm(test_loader, device)

    # ---- 3. 评估 STGNN ----
    stgnn_results = evaluate_stgnn(test_loader, edge_index, device)

    # ---- 4. 打印对比表格 ----
    comparison = print_comparison(lstm_results, stgnn_results)

    # ---- 5. 保存结果 ----
    save_results(comparison)

    print(f"\n🎉 TODO 4 完成！")
