# ============================================================
# experiments/a2_stability/train_multi_seed.py
# 多 seed 单次训练 + 评估脚本
# ============================================================
# 用法:
#   python experiments/a2_stability/train_multi_seed.py --preset A2B1 --seed 114514
#
# 与主线 train_basic_dynatopo.py 的区别：
#   - 模型初始化与训练随机性由 --seed 控制
#   - 验证集拆分固定用 seed=42（保证各 seed 的 val 指标可比）
#   - 模型输出到 experiments/models/，日志输出到 experiments/logs/
# ============================================================

import os
import sys
import json
import time
import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 项目根目录（experiments/a2_stability 的上两级）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from configs.config import (
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT
)
from configs.dynatopo_config import get_experiment_config
from core_models.stgnn_static import STGNN_Static
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from utils.loss_functions import CombinedLoss
from utils.metrics import compute_rmse, compute_nasa_score

# 实验区配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_seeds import NUM_EPOCHS, EARLY_STOP_PATIENCE, MODEL_DIR, LOG_DIR


# ============================================================
# 1. 数据加载（验证集拆分固定 seed=42，与主线一致）
# ============================================================
def load_data(subset='FD001', val_ratio=0.2):
    data = np.load(os.path.join(ROOT, f'data/processed/{subset}_train.npz'))
    X, y = data['X'], data['y']
    graph = torch.load(os.path.join(ROOT, f'data/processed/{subset}_train_graph.pt'))
    edge_index = graph['edge_index']

    # 固定用 RANDOM_SEED(42) 拆分，保证所有 seed 的验证集一致
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=RANDOM_SEED
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, edge_index, (X_val, y_val)


def load_test(subset='FD001'):
    data = np.load(os.path.join(ROOT, f'data/processed/{subset}_test.npz'))
    X_test = torch.tensor(data['X'], dtype=torch.float32)
    y_test = data['y']
    ruls = data.get('rul_true', y_test)
    graph = torch.load(os.path.join(ROOT, f'data/processed/{subset}_train_graph.pt'))
    edge_index = graph['edge_index']
    return X_test, y_test, ruls, edge_index


# ============================================================
# 2. 训练循环
# ============================================================
def train_one_epoch(model, loader, loss_fn, optimizer, edge_index, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch, edge_index.to(device))
        loss = loss_fn(pred, y_batch)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸（与 train_basic_static.py 保持一致）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, edge_index, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch, edge_index.to(device))
        loss = loss_fn(pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        all_preds.append(pred.cpu())
        all_labels.append(y_batch.cpu())
    return (total_loss / len(loader.dataset),
            torch.cat(all_preds), torch.cat(all_labels))


@torch.no_grad()
def evaluate_test(model, X_test, edge_index, device):
    model.eval()
    loader = DataLoader(TensorDataset(X_test, torch.zeros(len(X_test))),
                        batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    for X_batch, _ in loader:
        preds.append(model(X_batch.to(device), edge_index.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ============================================================
# 3. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='A1B1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ---- 设置本次训练的随机种子（模型初始化 + 训练过程）----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{args.preset}][seed={args.seed}] 设备: {device}")

    # ---- 加载数据 ----
    train_loader, val_loader, edge_index, (X_val, y_val) = load_data('FD001')
    X_test, y_test, ruls, edge_index_test = load_test('FD001')

    # ---- 构建模型 ----
    if args.preset == 'static':
        # 静态基线：使用 STGNN_Static（与主线 train_basic_static.py 一致）
        model = STGNN_Static(
            num_sensors=14, num_op_settings=3, use_transformer=False
        ).to(device)
    else:
        # 双图模型：使用 STGNN_DynaTopo（2×2 消融矩阵）
        cfg = get_experiment_config(args.preset)
        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3).to(device)

    # 防死亡ReLU：将 fc 最后一个 Linear 层的 bias 初始化为正数。
    # 若 bias 初始为负，最后一层 ReLU 输入恒为负 → 输出恒为 0，
    # 梯度无法回传（死亡ReLU），训练完全停滞（val_rmse 锁死在 RMS(y)≈90）。
    for layer in model.fc:
        if isinstance(layer, torch.nn.Linear):
            last_linear = layer
    last_linear.bias.data.fill_(1.0)

    loss_fn = CombinedLoss(MSE_WEIGHT, NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- 训练 ----
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     edge_index, device)
        val_loss, y_pred, y_true = validate(model, val_loader, loss_fn,
                                            edge_index, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    # ---- 恢复最佳权重 ----
    model.load_state_dict(best_state)

    # ---- 验证集指标（用最佳权重重算）----
    val_loss, y_pred, y_true = validate(model, val_loader, loss_fn,
                                        edge_index, device)
    val_rmse = compute_rmse(y_pred, y_true)
    val_nasa = compute_nasa_score(y_pred, y_true)

    # ---- 测试集指标 ----
    preds = evaluate_test(model, X_test, edge_index_test, device)
    test_rmse = compute_rmse(preds, y_test)
    test_nasa = compute_nasa_score(preds, ruls)
    params = sum(p.numel() for p in model.parameters())

    # ---- 保存模型与日志 ----
    os.makedirs(os.path.join(ROOT, MODEL_DIR), exist_ok=True)
    os.makedirs(os.path.join(ROOT, LOG_DIR), exist_ok=True)

    model_path = os.path.join(ROOT, MODEL_DIR, f'{args.preset}_seed{args.seed}_best_FD001.pt')
    torch.save({'model_state_dict': best_state, 'seed': args.seed,
                'preset': args.preset}, model_path)

    result = {
        'preset': args.preset,
        'seed': args.seed,
        'epochs_trained': epoch + 1,
        'val_loss': float(val_loss),
        'val_rmse': float(val_rmse),
        'val_nasa_score': float(val_nasa),
        'test_rmse': float(test_rmse),
        'test_nasa_score': float(test_nasa),
        'params': params,
    }
    log_path = os.path.join(ROOT, LOG_DIR, f'{args.preset}_seed{args.seed}.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[{args.preset}][seed={args.seed}] val_rmse={val_rmse:.2f} "
          f"val_nasa={val_nasa:.1f} test_rmse={test_rmse:.2f} "
          f"test_nasa={test_nasa:.1f} → {model_path}")


if __name__ == '__main__':
    main()
