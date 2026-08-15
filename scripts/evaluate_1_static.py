# ============================================================
# scripts/evaluate_1_static.py —— 单工况预测性能对比实验
# ============================================================
# TODO 4: 在 FD001 测试集上公平对比五种模型的预测性能:
#   LSTM / STGNN (MSTCN + GAT) / GRU / TCN / CNN+LSTM
# STGNN 采用基于消融实验结论的无 Transformer 变体
#
# 对比指标:
#   - RMSE（均方根误差，越低越好）
#   - NASA Score（C-MAPSS 官方非对称评分，越低越好）
#   - 参数量
#
# 用法:
#   python scripts/evaluate_1_static.py
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
from core_models.base_models import BasicLSTM, GRUModel, TCNModel, CNN_LSTM_Model
from core_models.stgnn_static import STGNN_Static
from utils.metrics import compute_rmse, compute_nasa_score, evaluate_metrics

# ============================================================
# 固定随机种子
# ============================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性推理（保证同模型可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
# 2. 各模型构建函数（超参数与对应训练脚本完全一致，保证公平对比）
# ============================================================
def build_lstm(device):
    """BasicLSTM —— 与 train_basic_lstm.py 一致"""
    return BasicLSTM(
        input_dim=NUM_FEATURES,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    ).to(device)


def build_gru(device):
    """GRUModel —— 与 train_basic_gru.py 一致（超参数与 LSTM 相同）"""
    return GRUModel(
        input_dim=NUM_FEATURES,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    ).to(device)


def build_tcn(device):
    """TCNModel —— 与 train_basic_tcn.py 一致"""
    return TCNModel(
        input_dim=NUM_FEATURES,
        num_channels=64,
        kernel_size=3,
        num_layers=4,
        dropout=0.3
    ).to(device)


def build_cnn_lstm(device):
    """CNN_LSTM_Model —— 与 train_basic_cnn_lstm.py 一致"""
    return CNN_LSTM_Model(
        input_dim=NUM_FEATURES,
        cnn_channels=64,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3
    ).to(device)


def build_stgnn(device):
    """STGNN_Static（MSTCN + GAT，无 Transformer）—— 与 train_basic_static.py 一致"""
    return STGNN_Static(
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


# ============================================================
# 3. 通用模型评估函数（所有模型共用一套推理流程）
# ============================================================
def evaluate_model(test_loader, device, model, model_name, model_path,
                   edge_index=None):
    """
    加载指定模型权重并在测试集上预测，统一计算指标

    参数:
        test_loader: 测试 DataLoader
        device:      设备
        model:       已实例化并放到 device 的模型
        model_name:  模型显示名称（用于日志输出）
        model_path:  权重文件路径
        edge_index:  图边索引（仅 STGNN 需要，其余模型传 None）

    返回:
        rmse, score, y_pred_all, y_true_all, num_params
    """
    print(f"\n{'='*60}")
    print(f"  🔍 评估 {model_name}")
    print(f"{'='*60}")

    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 {model_name} 模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  已加载权重 (Epoch {checkpoint['epoch']+1}, "
          f"Val Loss: {checkpoint['best_loss']:.4f})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")

    # 预测
    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if edge_index is not None:
                # STGNN 推理时需要传入 edge_index（forward 内部自动按 batch 扩展）
                y_pred = model(X_batch, edge_index.to(device))
            else:
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
# 4. 结果汇总与输出
# ============================================================
def print_comparison(results_list, baseline_name='LSTM', subset='FD001'):
    """
    打印多模型对比表格

    参数:
        results_list:  元素为 (model_name, rmse, score, num_params) 的列表
        baseline_name: 基准模型名（用于计算相对提升幅度）
        subset:        数据集名称

    返回:
        各模型指标字典（用于保存）
    """
    print(f"\n{'='*75}")
    print(f"  📊 {subset} 单工况预测性能对比")
    print(f"{'='*75}")
    print(f"  {'模型':<16}{'RMSE ↓':>10}{'NASA Score ↓':>14}{'参数量':>12}")
    print(f"  {'-'*75}")

    # 记录 LSTM 基准指标
    baseline_rmse = baseline_score = None
    for name, rmse, score, params in results_list:
        print(f"  {name:<16}{rmse:>10.2f}{score:>14.2f}{params:>12,}")
        if name == baseline_name:
            baseline_rmse, baseline_score = rmse, score

    # 找出 RMSE 最优模型
    best_name, best_rmse, best_score, _ = min(results_list, key=lambda r: r[1])

    print(f"  {'-'*75}")
    print(f"  🏆 RMSE 最优模型: {best_name}  (RMSE={best_rmse:.2f}, "
          f"NASA Score={best_score:.2f})")

    # 相对 LSTM 基准的提升幅度
    if baseline_rmse is not None:
        for name, rmse, score, params in results_list:
            if name == baseline_name:
                continue
            rmse_improve = (baseline_rmse - rmse) / baseline_rmse * 100
            score_improve = (baseline_score - score) / baseline_score * 100
            rmse_word = "✅ 降低" if rmse_improve >= 0 else "⚠️ 升高"
            score_word = "✅ 降低" if score_improve >= 0 else "⚠️ 升高"
            print(f"  · {name:<10} vs {baseline_name}: "
                  f"RMSE {rmse_word} {abs(rmse_improve):.1f}% | "
                  f"NASA Score {score_word} {abs(score_improve):.1f}%")
    print(f"{'='*75}")

    return {
        name: {
            'rmse': float(rmse),
            'score': float(score),
            'params': int(params)
        }
        for name, rmse, score, params in results_list
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
    print("=" * 75)
    print("  🧪 TODO 4: FD001 单工况预测性能对比实验")
    print("  LSTM vs STGNN vs GRU vs TCN vs CNN+LSTM")
    print("=" * 75)

    # ---- 设备选择 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  评估设备: {device}")

    # ---- 1. 加载测试数据（所有模型共用） ----
    test_loader, edge_index, y_test = load_test_data(subset='FD001')

    # ---- 2. 定义评估任务（模型名, 构建函数, 权重路径, edge_index） ----
    tasks = [
        ('LSTM',       build_lstm,    'saved_models/lstm_best_FD001.pt',         None),
        ('STGNN',      build_stgnn,   'saved_models/stgnn_static_best_FD001.pt', edge_index),
        ('GRU',        build_gru,     'saved_models/gru_best_FD001.pt',          None),
        ('TCN',        build_tcn,     'saved_models/tcn_best_FD001.pt',          None),
        ('CNN+LSTM',   build_cnn_lstm, 'saved_models/cnn_lstm_best_FD001.pt',    None),
    ]

    # ---- 3. 依次评估各模型 ----
    results_list = []
    for model_name, builder, model_path, edge in tasks:
        model = builder(device)
        rmse, score, _, _, num_params = evaluate_model(
            test_loader, device, model, model_name, model_path,
            edge_index=edge
        )
        results_list.append((model_name, rmse, score, num_params))

    # ---- 4. 打印对比表格 ----
    comparison = print_comparison(results_list)

    # ---- 5. 保存结果 ----
    save_results(comparison)

    print(f"\n🎉 TODO 4 完成！")
