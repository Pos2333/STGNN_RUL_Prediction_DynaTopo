# ============================================================
# scripts/ablation_dynatopo.py —— 双图模型消融实验
# ============================================================
# 对比 4 组变体:
#   1. 仅静态图（= 原 STGNN 基线）
#   2. 仅动态图（无 Spearman 先验）
#   3. 完整双图 A1B1（相似度+特征融合）
#   4. 完整双图 A2B1（注意力+特征融合）
#
# 用法:
#   python scripts/ablation_dynatopo.py
# ============================================================

import os
import sys
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, EARLY_STOP_PATIENCE, RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT
)
from configs.dynatopo_config import get_experiment_config
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from utils.loss_functions import CombinedLoss
from utils.metrics import compute_rmse, compute_nasa_score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data(subset='FD001', processed_dir='data/processed', val_ratio=0.2):
    train_path = os.path.join(processed_dir, f'{subset}_train.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')
    train_data = np.load(train_path)
    X, y = train_data['X'], train_data['y']
    graph = torch.load(graph_path)
    edge_index = graph['edge_index']

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
    return train_loader, val_loader, edge_index


def train_and_eval(model, train_loader, val_loader, edge_index, device,
                   num_epochs=50, patience=15):
    loss_fn = CombinedLoss(MSE_WEIGHT, NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_b, edge_index.to(device)), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b, edge_index.to(device))
                loss = loss_fn(pred, y_b)
                val_loss += loss.item() * X_b.size(0)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y_b.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            best_rmse = compute_rmse(preds, labels)
            best_score = compute_nasa_score(preds, labels)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_rmse, best_score, sum(p.numel() for p in model.parameters())


def main():
    print("=" * 60)
    print("  🔬 STGNN_DynaTopo 消融实验")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}\n")

    train_loader, val_loader, edge_index = load_data('FD001')

    # 定义消融变体
    variants = {
        "仅静态图（原STGNN基线）": "static_only",
        "仅动态图（无Spearman）": "dynamic_only",
        "完整双图 A1B1": "A1B1",
        "完整双图 A2B1": "A2B1",
    }

    results = {}

    for name, preset in variants.items():
        print(f"\n{'─'*50}")
        print(f"  🔍 {name} ({preset})")

        cfg = get_experiment_config(preset)
        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3).to(device)

        rmse, score, params = train_and_eval(
            model, train_loader, val_loader, edge_index, device
        )

        print(f"  ✅ RMSE: {rmse:.2f}, NASA Score: {score:.1f}, 参数: {params:,}")
        results[name] = {'rmse': float(rmse), 'score': float(score), 'params': params}

    # 打印汇总
    print(f"\n{'='*60}")
    print("  📋 消融实验汇总")
    print(f"{'='*60}")
    print(f"  {'模型变体':<30} {'RMSE':>8} {'NASA Score':>12} {'参数':>10}")
    print(f"  {'─'*60}")
    for name, r in results.items():
        print(f"  {name:<30} {r['rmse']:>8.2f} {r['score']:>12.1f} {r['params']:>10,}")

    # 保存
    os.makedirs('logs/dynatopo', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'logs/dynatopo/ablation_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📝 结果已保存至 logs/dynatopo/ablation_{ts}.json")


if __name__ == '__main__':
    main()
