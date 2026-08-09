# ============================================================
# scripts/ablation_study_v2.py —— 消融实验脚本（v2：基于无 Transformer 基线）
# ============================================================
# TODO 6 (v2): 在 MSTCN + GAT 新基线上做 2×2 消融，
# 验证两个保留模块的各自贡献，并以原始完整 STGNN 为参考
#
# 实验设计（五种变体在 FD001 上从零训练，FD001 测试集评估）:
#     1. MSTCN + GAT:          MSTCN ✅ + GAT ✅           (新基线)
#     2. 仅 GAT:               MSTCN ❌ + GAT ✅           (验证 MSTCN 贡献)
#     3. 仅 MSTCN:             MSTCN ✅ + GAT ❌           (验证 GAT 贡献)
#     4. 全关（最简模型）:      MSTCN ❌ + GAT ❌           (下限)
#     5. 完整 STGNN（参考）:    MSTCN ✅ + GAT ✅ + Trans ✅ (原始基线对比)
#
# 所有 v2 变体默认 use_transformer=False，
# 变体 5 的 use_transformer=True 仅在参考组中启用
#
# 输出:
#   - 终端打印 RMSE / NASA Score 对比表格
#   - 结果保存到 logs/ablation_v2_*.json
#   - 模型保存到 saved_models/ablation_v2_*.pt
#
# 用法:
#   python scripts/ablation_study_v2.py
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
# 消融实验配置定义（v2: 基于 MSTCN + GAT 新基线）
# ============================================================
ABLATION_CONFIGS = {
    # ---- 四个 v2 变体（use_transformer 统一为 False） ----
    "MSTCN + GAT (新基线)":  dict(use_mstcn=True,  use_gat=True,  use_transformer=False),
    "仅 GAT (无 MSTCN)":     dict(use_mstcn=False, use_gat=True,  use_transformer=False),
    "仅 MSTCN (无 GAT)":     dict(use_mstcn=True,  use_gat=False, use_transformer=False),
    "全关 (最简模型)":        dict(use_mstcn=False, use_gat=False, use_transformer=False),
    # ---- 原始完整版作为参考 ----
    "完整 STGNN (原始)":     dict(use_mstcn=True,  use_gat=True,  use_transformer=True),
}

# 预训练模型映射：这部分变体直接复用已有模型，无需从零训练
# （各脚本共用 RANDOM_SEED=42 + val_ratio=0.2，训练/验证集划分完全一致）
PRETRAINED_PATHS = {
    "MSTCN + GAT (新基线)": "saved_models/stgnn_v2_best_FD001.pt",
    "完整 STGNN (原始)":    "saved_models/stgnn_best_FD001.pt",
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

    graph = torch.load(graph_path, weights_only=False)
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
        name:         模型名称
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
# 4b. 根据配置构建模型（不训练）—— 供预训练加载使用
# ============================================================
def build_model_from_cfg(cfg, device):
    """仅构建模型结构，不训练"""
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
    return model


# ============================================================
# 5. 主函数：循环训练 + 评估 + 输出对比表
# ============================================================
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     STGNN 消融实验 v2 —— MSTCN + GAT 新基线分析    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ---- 加载数据 ----
    train_loader, val_loader, test_loader, edge_index = load_data_and_graph(
        subset='FD001'
    )

    # ---- 逐变体训练/加载 + 评估 ----
    results = {}

    for name, cfg in ABLATION_CONFIGS.items():
        pretrain_path = PRETRAINED_PATHS.get(name)

        if pretrain_path and os.path.exists(pretrain_path):
            # ---- 直接加载预训练模型，跳过训练 ----
            print(f"\n{'='*60}")
            print(f"  🔬 变体: {name}")
            print(f"     use_mstcn={cfg['use_mstcn']}, "
                  f"use_gat={cfg['use_gat']}, "
                  f"use_transformer={cfg['use_transformer']}")
            print(f"  📥 直接加载预训练模型: {pretrain_path}")
            print(f"{'='*60}")

            model = build_model_from_cfg(cfg, DEVICE)
            ckpt = torch.load(pretrain_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  已加载权重 (Epoch {ckpt.get('epoch', '?')}, "
                  f"Val Loss: {ckpt.get('best_loss', ckpt.get('best_tgt_val_loss', '?'))})")
        elif pretrain_path and not os.path.exists(pretrain_path):
            # 预训练模型不存在，回退到训练
            print(f"\n  ⚠️  预训练模型 {pretrain_path} 不存在，将从零训练")
            model, _ = train_ablation_model(
                name, cfg, train_loader, val_loader, edge_index, DEVICE
            )
        else:
            # 从零训练
            model, _ = train_ablation_model(
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

        # ---- 保存模型（预训练复用的跳过，避免冗余） ----
        if name not in PRETRAINED_PATHS:
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
            model_path = f"saved_models/ablation_v2_{safe_name}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  💾 模型已保存 → {model_path}")
        print(f"     📊 {name}: RMSE={rmse:.4f}, Score={score:.4f}")

    # ============================================================
    # 6. 输出对比表格
    # ============================================================
    # 新基线 = "MSTCN + GAT (新基线)"
    base_name = "MSTCN + GAT (新基线)"
    base_rmse = results[base_name]['rmse']
    base_score = results[base_name]['score']

    print(f"\n{'='*75}")
    print(f"  📊 消融实验 v2 最终结果对比")
    print(f"  新基线: {base_name}  |  RMSE={base_rmse:.4f}, Score={base_score:.4f}")
    print(f"{'='*75}")
    print(f"  {'模型变体':<30s} {'RMSE':>10s} {'NASA Score':>14s} {'参数量':>12s}")
    print(f"  {'-'*66}")

    for name, r in results.items():
        marker = " ← 基线" if name == base_name else ""
        print(f"  {name:<30s} {r['rmse']:>10.4f} {r['score']:>14.4f} {r['params']:>12,}{marker}")

    # ---- 相对于新基线的退化分析 ----
    print(f"\n  📈 相对于「{base_name}」的性能退化:")
    for name, r in results.items():
        if name == base_name:
            continue
        rmse_pct = (r['rmse'] - base_rmse) / base_rmse * 100
        score_pct = (r['score'] - base_score) / base_score * 100
        print(f"     {name:<30s}: RMSE {rmse_pct:+.1f}%,  Score {score_pct:+.1f}%")

    # ---- 2×2 交叉分析（V1-V4） ----
    v2_keys = ["MSTCN + GAT (新基线)", "仅 GAT (无 MSTCN)",
               "仅 MSTCN (无 GAT)", "全关 (最简模型)"]
    print(f"\n  📐 2×2 交叉分析（MSTCN × GAT）:")
    if all(k in results for k in v2_keys):
        r = {k: results[k] for k in v2_keys}
        # MSTCN 主效应: (MSTCN+GAT + MSTCN) vs (GAT + 全关)
        mstcn_on_rmse = (r["MSTCN + GAT (新基线)"]['rmse'] + r["仅 MSTCN (无 GAT)"]['rmse']) / 2
        mstcn_off_rmse = (r["仅 GAT (无 MSTCN)"]['rmse'] + r["全关 (最简模型)"]['rmse']) / 2
        # GAT 主效应
        gat_on_rmse = (r["MSTCN + GAT (新基线)"]['rmse'] + r["仅 GAT (无 MSTCN)"]['rmse']) / 2
        gat_off_rmse = (r["仅 MSTCN (无 GAT)"]['rmse'] + r["全关 (最简模型)"]['rmse']) / 2

        print(f"     MSTCN 主效应 (RMSE): 启用={mstcn_on_rmse:.2f}, 关闭={mstcn_off_rmse:.2f}, Δ={mstcn_off_rmse - mstcn_on_rmse:+.2f}")
        print(f"     GAT 主效应   (RMSE): 启用={gat_on_rmse:.2f}, 关闭={gat_off_rmse:.2f}, Δ={gat_off_rmse - gat_on_rmse:+.2f}")

    print(f"{'='*75}")

    # ---- 保存结果到 JSON ----
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"logs/ablation_v2_{timestamp}.json"
    os.makedirs('logs', exist_ok=True)

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
        'baseline_name': base_name,
        'base_rmse': base_rmse,
        'base_score': base_score,
        'device': str(DEVICE),
        'num_epochs': NUM_EPOCHS,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'note': 'v2 消融实验：MSTCN+GAT 新基线上的 2×2 交叉验证 + 原始完整版参考',
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
