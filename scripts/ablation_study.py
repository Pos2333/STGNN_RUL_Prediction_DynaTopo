# ============================================================
# scripts/ablation_study.py —— 消融实验脚本
# ============================================================
# TODO 6: 逐一拆除 STGNN 核心模块，验证各组件对性能的贡献
#
# 实验设计:
#   四种模型变体在 FD001 上从零训练，在 FD001 测试集上评估:
#     1. 完整 STGNN:        MSTCN ✅ + GAT ✅ + Transformer ✅
#     2. 无 MSTCN:          简单Conv替代 + GAT ✅ + Transformer ✅
#     3. 无 GAT:            MSTCN ✅ + 均值池化替代 + Transformer ✅
#     4. 无 Transformer:    MSTCN ✅ + GAT ✅ + 均值池化替代
#
# 输出:
#   - 终端打印 RMSE / NASA Score 对比表格
#   - 每轮结果保存到 logs/ablation_*.json
#   - 每个变体的最佳模型保存到 saved_models/ablation_*.pt
#
# 用法:
#   python scripts/ablation_study.py
# ============================================================

import os
import sys
import json
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
    RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from core_models.stgnn_full import STGNN
from utils.loss_functions import CombinedLoss
from utils.metrics import evaluate_metrics

# ============================================================
# 固定随机种子，保证公平对比
# ============================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 设备选择
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  设备: {DEVICE}")

# ============================================================
# 消融实验配置定义
# ============================================================
ABLATION_CONFIGS = {
    "完整 STGNN":        dict(use_mstcn=True,  use_gat=True,  use_transformer=True),
    "无 MSTCN":          dict(use_mstcn=False, use_gat=True,  use_transformer=True),
    "无 GAT":            dict(use_mstcn=True,  use_gat=False, use_transformer=True),
    "无 Transformer":    dict(use_mstcn=True,  use_gat=True,  use_transformer=False),
}


# ============================================================
# 1. 加载数据 + 图结构
# ============================================================
def load_data_and_graph(subset='FD001', processed_dir='data/processed',
                        val_ratio=0.2):
    """
    加载预处理数据，切分训练/验证集

    返回:
        train_loader, val_loader, edge_index, test_loader
    """
    train_path = os.path.join(processed_dir, f'{subset}_train.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')
    test_path = os.path.join(processed_dir, f'{subset}_test.npz')

    for p, name in [(train_path, '训练'), (graph_path, '图结构'), (test_path, '测试')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到{name}数据: {p}")

    # ---- 加载训练数据 ----
    train_data = np.load(train_path)
    X = train_data['X']
    y = train_data['y']

    graph = torch.load(graph_path)
    edge_index = graph['edge_index']

    print(f"\n📂 数据加载: {subset}")
    print(f"  样本数: {len(X)}, 特征形状: {X.shape[1:]}")
    print(f"  图边数: {edge_index.shape[1]}")

    # ---- 切分训练/验证集 ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=RANDOM_SEED, shuffle=True
    )
    print(f"  训练: {len(X_train)}, 验证: {len(X_val)}")

    # ---- 加载测试数据 ----
    test_data = np.load(test_path)
    X_test = test_data['X']
    y_test = test_data['y']
    print(f"  测试: {len(X_test)}")

    # ---- 转 tensor ----
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # ---- DataLoader ----
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader, edge_index


# ============================================================
# 2. 训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, loss_fn, optimizer, edge_index, device):
    """执行一个训练 epoch"""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        edge_device = edge_index.to(device) if edge_index is not None else None

        y_pred = model(X_batch, edge_device)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 3. 验证/测试
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, loss_fn, edge_index, device):
    """在验证集或测试集上评估"""
    model.eval()
    total_loss = 0.0
    y_pred_all, y_true_all = [], []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        edge_device = edge_index.to(device) if edge_index is not None else None

        y_pred = model(X_batch, edge_device)
        loss = loss_fn(y_pred, y_batch)

        total_loss += loss.item()
        y_pred_all.append(y_pred.cpu().numpy())
        y_true_all.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    return avg_loss, y_pred_all, y_true_all


# ============================================================
# 4. 训练单个模型变体
# ============================================================
def train_ablation_model(name, cfg, train_loader, val_loader, edge_index, device):
    """
    训练一个消融模型变体

    参数:
        name:         模型名称（如"完整 STGNN"）
        cfg:          消融开关配置字典
        train_loader: 训练 DataLoader
        val_loader:   验证 DataLoader
        edge_index:   图边索引（use_gat=False 的变体不使用）
        device:       训练设备

    返回:
        model:        训练好的最佳模型
        best_val_loss: 最佳验证损失
    """
    print(f"\n{'='*60}")
    print(f"  🔬 训练变体: {name}")
    print(f"     use_mstcn={cfg['use_mstcn']}, "
          f"use_gat={cfg['use_gat']}, "
          f"use_transformer={cfg['use_transformer']}")
    print(f"{'='*60}")

    # ---- 构建模型 ----
    model = STGNN(
        num_sensors=14, num_op_settings=3,
        mstcn_channels=MSTCN_NUM_CHANNELS, mstcn_kernels=MSTCN_KERNEL_SIZES,
        mstcn_dropout=MSTCN_DROPOUT,
        gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_HEADS, gat_dropout=GAT_DROPOUT,
        trans_d_model=TRANSFORMER_D_MODEL, trans_nhead=TRANSFORMER_NHEAD,
        trans_num_layers=TRANSFORMER_NUM_LAYERS, trans_dropout=TRANSFORMER_DROPOUT,
        use_mstcn=cfg['use_mstcn'], use_gat=cfg['use_gat'],
        use_transformer=cfg['use_transformer'],
        fc_hidden=FC_HIDDEN_DIM
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,}")

    # ---- 损失函数 & 优化器 ----
    loss_fn = CombinedLoss(
        mse_weight=MSE_WEIGHT,
        nasa_weight=NASA_SCORE_WEIGHT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- 训练设置 ----
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses, val_losses = [], []

    # GAT 关闭时不需要 edge_index
    train_edge = edge_index if cfg['use_gat'] else None

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, train_edge, device
        )

        # 验证
        val_loss, _, _ = evaluate(
            model, val_loader, loss_fn, train_edge, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        # ---- 早停判断 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # 每 10 轮或首尾轮打印
        if epoch % 10 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"⏱ {epoch_time:.1f}s"
                  + (" ⭐ 最佳" if patience_counter == 0 else ""))

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  ⏹ 早停于第 {epoch+1} 轮（{EARLY_STOP_PATIENCE} 轮未改善）")
            break

    # ---- 恢复最佳模型 ----
    model.load_state_dict(best_model_state)

    return model, best_val_loss


# ============================================================
# 5. 主函数：循环训练 + 评估 + 输出对比表
# ============================================================
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║        STGNN 消融实验 —— 组件贡献度分析             ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ---- 加载数据 ----
    train_loader, val_loader, test_loader, edge_index = load_data_and_graph(
        subset='FD001'
    )

    # ---- 逐变体训练 + 评估 ----
    results = {}

    for name, cfg in ABLATION_CONFIGS.items():
        # 训练
        model, best_val_loss = train_ablation_model(
            name, cfg, train_loader, val_loader, edge_index, DEVICE
        )

        # 测试集评估
        test_edge = edge_index if cfg['use_gat'] else None
        test_loss, y_pred, y_true = evaluate(
            model, test_loader, CombinedLoss(MSE_WEIGHT, NASA_SCORE_WEIGHT),
            test_edge, DEVICE
        )
        rmse, score = evaluate_metrics(y_pred, y_true, print_result=False)

        results[name] = {
            'rmse': round(float(rmse), 4),
            'score': round(float(score), 4),
            'params': sum(p.numel() for p in model.parameters()),
            'cfg': cfg,
        }

        # ---- 保存模型 ----
        model_path = f"saved_models/ablation_{name.replace(' ', '_')}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  💾 模型已保存 → {model_path}")
        print(f"     📊 {name}: RMSE={rmse:.4f}, Score={score:.4f}")

    # ============================================================
    # 6. 输出对比表格
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  📊 消融实验最终结果对比")
    print(f"{'='*70}")
    print(f"  {'模型变体':<20s} {'RMSE':>10s} {'NASA Score':>12s} {'参数量':>12s}")
    print(f"  {'-'*54}")

    # 以完整 STGNN 为基准计算退化比例
    base_rmse = results["完整 STGNN"]['rmse']
    base_score = results["完整 STGNN"]['score']

    for name, r in results.items():
        rmse_delta = r['rmse'] - base_rmse
        score_delta = r['score'] - base_score
        print(f"  {name:<20s} {r['rmse']:>10.4f} {r['score']:>12.4f} {r['params']:>12,}")

    print(f"\n  📈 相对于完整 STGNN 的性能退化:")
    for name, r in results.items():
        if name == "完整 STGNN":
            continue
        rmse_pct = (r['rmse'] - base_rmse) / base_rmse * 100
        score_pct = (r['score'] - base_score) / base_score * 100
        print(f"     {name:<20s}: RMSE +{rmse_pct:+.1f}%,  Score +{score_pct:+.1f}%")

    print(f"{'='*70}")

    # ---- 保存结果到 JSON ----
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"logs/ablation_{timestamp}.json"
    os.makedirs('logs', exist_ok=True)

    # 转换结果以便 JSON 序列化
    json_results = {}
    for name, r in results.items():
        json_results[name] = {
            'rmse': r['rmse'],
            'score': r['score'],
            'params': r['params'],
            'cfg': r['cfg'],
        }
    json_results['meta'] = {
        'timestamp': timestamp,
        'base_rmse': base_rmse,
        'base_score': base_score,
        'device': str(DEVICE),
        'num_epochs': NUM_EPOCHS,
        'early_stop_patience': EARLY_STOP_PATIENCE,
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n📝 结果已保存 → {log_path}")

    return results


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    main()
