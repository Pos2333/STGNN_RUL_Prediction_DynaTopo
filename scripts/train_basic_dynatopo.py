# ============================================================
# scripts/train_basic_dynatopo.py —— 双图 STGNN 训练脚本
# ============================================================
# 配置驱动的训练脚本，通过 --preset 参数切换 A×B 组合。
#
# 用法:
#   python scripts/train_basic_dynatopo.py --preset A1B1
#   python scripts/train_basic_dynatopo.py --preset A2B2
#   python scripts/train_basic_dynatopo.py --preset A1B1 --resume
#   python scripts/train_basic_dynatopo.py --list-presets
# ============================================================

import os
import sys
import json
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
    RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT
)
from configs.dynatopo_config import (
    get_experiment_config, list_all_presets,
    EXPERIMENT_MATRIX, ABLATION_CONFIGS
)
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from utils.loss_functions import CombinedLoss
from utils.metrics import evaluate_metrics, compute_rmse, compute_nasa_score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性训练（保证同种子可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 1. 数据加载（复用静态版本的逻辑）
# ============================================================
def load_data_and_graph(subset='FD001', processed_dir='data/processed', val_ratio=0.2):
    train_path = os.path.join(processed_dir, f'{subset}_train.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练数据: {train_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"找不到图结构: {graph_path}")

    train_data = np.load(train_path)
    X = train_data['X']
    y = train_data['y']
    graph = torch.load(graph_path)
    edge_index = graph['edge_index']

    print(f"\n📂 数据加载完成 - {subset}")
    print(f"  总样本数: {len(X)}, 特征形状: {X.shape[1:]}")
    print(f"  图边数: {edge_index.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=RANDOM_SEED, shuffle=True
    )
    print(f"  训练样本: {len(X_train)}, 验证样本: {len(X_val)}")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, edge_index


# ============================================================
# 2. 训练/验证辅助函数
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


def save_checkpoint(model, optimizer, epoch, best_loss,
                    train_losses, val_losses, filepath, preset):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch, 'best_loss': best_loss,
        'train_losses': train_losses, 'val_losses': val_losses,
        'preset': preset,
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    ckpt = torch.load(filepath, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_loss'], \
        ckpt.get('train_losses', []), ckpt.get('val_losses', [])


def train(model, train_loader, val_loader, loss_fn, optimizer, edge_index,
          device, preset, num_epochs=NUM_EPOCHS, patience=EARLY_STOP_PATIENCE,
          resume=False, checkpoint_path=None):
    if resume and os.path.exists(checkpoint_path):
        start_epoch, best_loss, train_losses, val_losses = load_checkpoint(
            model, optimizer, checkpoint_path)
        print(f"  📂 从 checkpoint 恢复: Epoch {start_epoch}")
    else:
        start_epoch, best_loss = 0, float('inf')
        train_losses, val_losses = [], []
    val_rmses, val_nasa_scores = [], []
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  🚀 开始训练 STGNN_DynaTopo [{preset}]")
    print(f"  设备: {device}, 最大轮数: {num_epochs}, 早停: {patience}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, edge_index, device)
        val_loss, y_pred, y_true = validate(model, val_loader, loss_fn, edge_index, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # 记录独立的 val RMSE 和 val NASA Score（不受 CombinedLoss 噪声影响）
        val_rmse = compute_rmse(y_pred, y_true)
        val_nasa = compute_nasa_score(y_pred, y_true)
        val_rmses.append(float(val_rmse))
        val_nasa_scores.append(float(val_nasa))
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"vRMSE: {val_rmse:.2f} | vNASA: {val_nasa:.1f} | {epoch_time:.1f}s")
        else:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | {epoch_time:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_rmse = val_rmse
            best_nasa = val_nasa
            patience_counter = 0
            best_path = f'saved_models/dynatopo_{preset}_best_FD001.pt'
            torch.save({'model_state_dict': model.state_dict(),
                        'best_loss': best_loss, 'epoch': epoch, 'preset': preset,
                        'best_val_rmse': float(best_rmse), 'best_val_nasa_score': float(best_nasa)}, best_path)
            print(f"  ⭐ 新最佳模型！vRMSE={best_rmse:.2f} vNASA={best_nasa:.1f} → {best_path}")
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch, best_loss, train_losses, val_losses,
                        checkpoint_path, preset)

        if patience_counter >= patience:
            print(f"\n  🛑 早停！连续 {patience} 轮未改善")
            break

    print(f"\n  ✅ 训练完成！最佳 Val Loss: {best_loss:.4f}")
    return model, train_losses, val_losses, val_rmses, val_nasa_scores


def save_log(train_losses, val_losses, val_rmses, val_nasa_scores, preset, subset='FD001'):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/dynatopo/{preset}_{subset}_{timestamp}.json'
    os.makedirs('logs/dynatopo', exist_ok=True)
    best_idx = val_losses.index(min(val_losses)) if val_losses else 0
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': f'STGNN_DynaTopo_{preset}',
            'subset': subset,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_rmses': val_rmses,
            'val_nasa_scores': val_nasa_scores,
            'best_val_loss': min(val_losses) if val_losses else None,
            'best_val_rmse': val_rmses[best_idx] if val_rmses else None,
            'best_val_nasa_score': val_nasa_scores[best_idx] if val_nasa_scores else None,
            'num_epochs': len(train_losses),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📝 日志已保存 → {log_path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STGNN_DynaTopo 训练脚本')
    parser.add_argument('--preset', type=str, default='A1B1',
                        help='实验预设: A1B1, A1B2, A2B1, A2B2, static_only, dynamic_only')
    parser.add_argument('--resume', action='store_true', help='从 checkpoint 恢复训练')
    parser.add_argument('--list-presets', action='store_true', help='列出所有可用预设')
    args = parser.parse_args()

    if args.list_presets:
        list_all_presets()
        sys.exit(0)

    print("=" * 60)
    print(f"  🧪 STGNN_DynaTopo 训练 —— {args.preset}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 加载配置
    cfg = get_experiment_config(args.preset)
    print(f"\n📋 实验配置: {cfg.name}")
    print(f"  generator={cfg.generator}, fusion={cfg.fusion}, "
          f"use_static={cfg.use_static_graph}")

    # 加载数据
    train_loader, val_loader, edge_index = load_data_and_graph('FD001')

    # 创建模型
    model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3).to(device)
    print(f"\n🔧 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = f'saved_models/dynatopo_{args.preset}_checkpoint.pt'

    model, train_losses, val_losses, val_rmses, val_nasa_scores = train(
        model, train_loader, val_loader, loss_fn, optimizer, edge_index,
        device, args.preset,
        num_epochs=NUM_EPOCHS, patience=EARLY_STOP_PATIENCE,
        resume=args.resume, checkpoint_path=checkpoint_path
    )

    save_log(train_losses, val_losses, val_rmses, val_nasa_scores, args.preset)

    # 加载最佳模型
    best_path = f'saved_models/dynatopo_{args.preset}_best_FD001.pt'
    best_ckpt = torch.load(best_path, map_location='cpu')
    model.load_state_dict(best_ckpt['model_state_dict'])
    print(f"\n📊 验证集最终评估:")
    val_loss, y_pred, y_true = validate(model, val_loader, loss_fn, edge_index, device)
    evaluate_metrics(y_pred, y_true, print_result=True)

    print(f"\n✅ 训练完成！模型: saved_models/dynatopo_{args.preset}_best_FD001.pt")
