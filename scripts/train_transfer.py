# ============================================================
# scripts/train_transfer.py —— 跨工况迁移学习训练脚本
# ============================================================
# TODO 5 动作1：基于 FD001 预训练模型，迁移到 FD002
#
# 策略：
#   1. 加载 FD001 上训练好的 STGNN 作为预训练起点
#   2. 同时加载 FD001（源域）和 FD002（目标域）数据
#   3. 源域计算 CombinedLoss（MSE + NASA Score）
#   4. 对源域和目标域的中间特征计算 LMMD 损失
#   5. 总损失 = CombinedLoss + LMMD_LAMBDA * LMMD
#   6. 支持 checkpoint 暂停恢复
#
# 用法：
#   python scripts/train_transfer.py
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
    RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT, LMMD_LAMBDA, TGT_TASK_WEIGHT,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from core_models.stgnn_full import STGNN
from utils.loss_functions import CombinedLoss, lmmd_loss
from utils.metrics import evaluate_metrics

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1. 加载源域和目标域数据
# ============================================================
def load_transfer_data(source_subset='FD001', target_subset='FD002',
                       processed_dir='data/processed', val_ratio=0.2):
    """
    加载源域和目标域的训练数据，并从源域切出验证集

    参数:
        source_subset: 源域数据集编号
        target_subset: 目标域数据集编号
        processed_dir: 预处理数据目录
        val_ratio:     验证集比例

    返回:
        source_loader, target_loader, val_loader
        source_edge, target_edge
    """
    # ---- 加载源域数据（FD001） ----
    src_path = os.path.join(processed_dir, f'{source_subset}_train.npz')
    src_graph_path = os.path.join(processed_dir, f'{source_subset}_train_graph.pt')
    src_data = np.load(src_path)
    X_src = src_data['X']
    y_src = src_data['y']
    src_graph = torch.load(src_graph_path)
    src_edge = src_graph['edge_index']

    print(f"\n📂 源域 {source_subset}: {len(X_src)} 个训练样本")
    print(f"  源域图边数: {src_edge.shape[1]}")

    # ---- 加载目标域数据（FD002） ----
    tgt_path = os.path.join(processed_dir, f'{target_subset}_train.npz')
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"找不到目标域数据: {tgt_path}")
    tgt_data = np.load(tgt_path)
    X_tgt = tgt_data['X']
    y_tgt = tgt_data['y']

    print(f"📂 目标域 {target_subset}: {len(X_tgt)} 个训练样本")

    # ---- 目标域的图结构（可能不同于源域） ----
    tgt_graph_path = os.path.join(processed_dir, f'{target_subset}_train_graph.pt')
    if os.path.exists(tgt_graph_path):
        tgt_graph = torch.load(tgt_graph_path)
        tgt_edge = tgt_graph['edge_index']
        print(f"  目标域图边数: {tgt_edge.shape[1]}")
    else:
        # 如果目标域没有自己的图结构，使用源域的（跨工况共用图）
        tgt_edge = src_edge
        print(f"  目标域使用源域图结构（{tgt_edge.shape[1]} 边）")

    # ---- 源域拆出验证集 ----
    X_src_train, X_src_val, y_src_train, y_src_val = train_test_split(
        X_src, y_src, test_size=val_ratio, random_state=RANDOM_SEED, shuffle=True
    )
    print(f"  源域训练: {len(X_src_train)}, 源域验证: {len(X_src_val)}")

    # ---- 转为 tensor ----
    X_src_train = torch.tensor(X_src_train, dtype=torch.float32)
    y_src_train = torch.tensor(y_src_train, dtype=torch.float32).view(-1, 1)
    X_src_val = torch.tensor(X_src_val, dtype=torch.float32)
    y_src_val = torch.tensor(y_src_val, dtype=torch.float32).view(-1, 1)
    X_tgt = torch.tensor(X_tgt, dtype=torch.float32)
    y_tgt = torch.tensor(y_tgt, dtype=torch.float32).view(-1, 1)

    # ---- DataLoader ----
    src_dataset = TensorDataset(X_src_train, y_src_train)
    val_dataset = TensorDataset(X_src_val, y_src_val)
    tgt_dataset = TensorDataset(X_tgt, y_tgt)

    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, drop_last=False)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=False)

    print(f"📦 源域批次: {len(src_loader)}, 目标域批次: {len(tgt_loader)}, "
          f"验证批次: {len(val_loader)}")

    return src_loader, tgt_loader, val_loader, src_edge, tgt_edge


# ============================================================
# 2. 保存 / 加载 checkpoint
# ============================================================
def save_checkpoint(model, optimizer, epoch, best_loss,
                    train_losses, val_losses,
                    filepath='saved_models/transfer_checkpoint.pt'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, filepath)
    print(f"  💾 Checkpoint 已保存 → {filepath}")


def load_checkpoint(model, optimizer, filepath='saved_models/transfer_checkpoint.pt'):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"  🔄 从 Checkpoint 恢复，从第 {start_epoch + 1} 轮继续")
        return start_epoch, best_loss, train_losses, val_losses
    else:
        print("  🆕 未找到 Checkpoint，从头开始训练")
        return 0, float('inf'), [], []


# ============================================================
# 3. 训练一个 epoch（含 LMMD 迁移损失）
# ============================================================
def train_one_epoch_transfer(model, src_loader, tgt_loader,
                             task_loss_fn, src_edge, tgt_edge, optimizer,
                             device, lmmd_lambda=LMMD_LAMBDA,
                             tgt_task_weight=TGT_TASK_WEIGHT):
    """
    半监督迁移学习训练：源域 + 目标域都做监督学习，同时 LMMD 对齐特征

    训练策略：
      - 从源域和目标域各取一个 batch
      - 源域 batch: 计算任务损失（MSE + NASA）
      - 目标域 batch: 同样计算任务损失（利用 C-MAPSS 提供的标签）
      - 同时对两个域的中间特征计算 LMMD 进行子域对齐
      - 总损失 = 源域任务损失 + w × 目标域任务损失 + λ × LMMD

    参数:
        model:           STGNN 模型
        src_loader:      源域 DataLoader
        tgt_loader:      目标域 DataLoader
        task_loss_fn:    任务损失函数（CombinedLoss）
        src_edge:        源域图结构
        tgt_edge:        目标域图结构
        optimizer:       优化器
        device:          设备
        lmmd_lambda:     LMMD 损失权重
        tgt_task_weight: 目标域任务损失权重（0 则退化为无监督 LMMD）

    返回:
        avg_task_loss_src, avg_task_loss_tgt, avg_lmmd_loss, avg_total_loss
    """
    model.train()
    total_task_loss_src = 0.0
    total_task_loss_tgt = 0.0
    total_lmmd_loss = 0.0
    total_loss_sum = 0.0

    # 以较大的 DataLoader 长度为准进行迭代
    num_batches = max(len(src_loader), len(tgt_loader))
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    for _ in range(num_batches):
        # ---- 取源域 batch ----
        try:
            X_src, y_src = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            X_src, y_src = next(src_iter)

        # ---- 取目标域 batch（半监督：保留标签用于监督学习） ----
        try:
            X_tgt, y_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            X_tgt, y_tgt = next(tgt_iter)

        X_src = X_src.to(device)
        y_src = y_src.to(device)
        X_tgt = X_tgt.to(device)
        y_tgt = y_tgt.to(device)
        src_edge_d = src_edge.to(device)
        tgt_edge_d = tgt_edge.to(device)

        # ---- 源域前向（需要预测值用于任务损失 + 特征用于 LMMD） ----
        y_pred_src, feat_src = model(X_src, src_edge_d, return_feat=True)

        # ---- 目标域前向（同样需要预测值 + 特征） ----
        y_pred_tgt, feat_tgt = model(X_tgt, tgt_edge_d, return_feat=True)

        # ---- 源域任务损失 ----
        task_loss_src = task_loss_fn(y_pred_src, y_src)

        # ---- 目标域任务损失（利用 C-MAPSS 提供的标签做监督学习） ----
        task_loss_tgt = task_loss_fn(y_pred_tgt, y_tgt)

        # ---- 总任务损失 ----
        task_loss = task_loss_src + tgt_task_weight * task_loss_tgt

        # ---- LMMD 损失（对齐源域和目标域的特征分布） ----
        lmmd_val = lmmd_loss(feat_src, feat_tgt, y_src)

        # ---- 总损失 ----
        total_loss = task_loss + lmmd_lambda * lmmd_val

        # ---- 反向传播 ----
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_task_loss_src += task_loss_src.item()
        total_task_loss_tgt += task_loss_tgt.item()
        total_lmmd_loss += lmmd_val.item()
        total_loss_sum += total_loss.item()

    avg_task_src = total_task_loss_src / num_batches
    avg_task_tgt = total_task_loss_tgt / num_batches
    avg_lmmd = total_lmmd_loss / num_batches
    avg_total = total_loss_sum / num_batches
    return avg_task_src, avg_task_tgt, avg_lmmd, avg_total


# ============================================================
# 4. 验证（仅在源域验证集上）
# ============================================================
def validate_transfer(model, val_loader, task_loss_fn, src_edge, device):
    """在源域验证集上评估（目标域无标签，无法直接评估）"""
    model.eval()
    total_loss = 0.0
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            edge_d = src_edge.to(device)

            y_pred = model(X_batch, edge_d)
            loss = task_loss_fn(y_pred, y_batch)

            total_loss += loss.item()
            y_pred_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    return avg_loss, y_pred_all, y_true_all


# ============================================================
# 5. 主训练函数
# ============================================================
def train_transfer(model, src_loader, tgt_loader, val_loader,
                   task_loss_fn, src_edge, tgt_edge, optimizer, device,
                   target_subset='FD002',
                   num_epochs=NUM_EPOCHS, patience=EARLY_STOP_PATIENCE,
                   lmmd_lambda=LMMD_LAMBDA,
                   resume=False,
                   checkpoint_path='saved_models/transfer_checkpoint.pt'):
    """迁移学习主训练循环"""
    if resume:
        start_epoch, best_loss, train_losses, val_losses = load_checkpoint(
            model, optimizer, checkpoint_path
        )
    else:
        start_epoch = 0
        best_loss = float('inf')
        train_losses = []
        val_losses = []

    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  🚀 开始半监督迁移学习训练")
    print(f"  设备: {device}, 最大轮数: {num_epochs}, 早停: {patience}")
    print(f"  LMMD 权重 λ: {lmmd_lambda}, 目标域任务权重 w: {TGT_TASK_WEIGHT}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 迁移训练 ----
        task_loss_src, task_loss_tgt, lmmd_val, total_loss = train_one_epoch_transfer(
            model, src_loader, tgt_loader, task_loss_fn,
            src_edge, tgt_edge, optimizer, device, lmmd_lambda, TGT_TASK_WEIGHT
        )

        # ---- 验证 ----
        val_loss, y_pred, y_true = validate_transfer(
            model, val_loader, task_loss_fn, src_edge, device
        )

        train_losses.append(task_loss_src + task_loss_tgt)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            rmse, score = evaluate_metrics(y_pred, y_true, print_result=False)
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Src: {task_loss_src:.2f} | Tgt: {task_loss_tgt:.2f} | "
                  f"LMMD: {lmmd_val:.4f} | "
                  f"Val: {val_loss:.4f} | RMSE: {rmse:.2f} | Score: {score:.1f} | "
                  f"耗时: {epoch_time:.1f}s")
        else:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Src: {task_loss_src:.2f} | Tgt: {task_loss_tgt:.2f} | "
                  f"LMMD: {lmmd_val:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"耗时: {epoch_time:.1f}s")

        # ---- 保存最佳模型 ----
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_path = f'saved_models/transfer_best_{target_subset}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch,
            }, best_model_path)
            print(f"  ⭐ 新的最佳模型！Val Loss: {best_loss:.4f} → {best_model_path}")
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch, best_loss,
                        train_losses, val_losses, checkpoint_path)

        if patience_counter >= patience:
            print(f"\n  🛑 早停触发！验证 loss 连续 {patience} 轮未改善")
            break

    print(f"\n{'='*60}")
    print(f"  ✅ 迁移学习训练完成！最佳验证损失: {best_loss:.4f}")
    print(f"{'='*60}")

    return model, train_losses, val_losses


# ============================================================
# 6. 保存训练日志
# ============================================================
def save_training_log(train_losses, val_losses, source='FD001', target='FD002'):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/transfer_{source}_to_{target}_{timestamp}.json'

    log_data = {
        'model': 'STGNN_Transfer',
        'source': source,
        'target': target,
        'lmmd_lambda': LMMD_LAMBDA,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses) if val_losses else None,
        'num_epochs': len(train_losses),
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"\n📝 训练日志已保存 → {log_path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    # ---- 命令行参数 ----
    import argparse
    parser = argparse.ArgumentParser(description='跨工况迁移学习训练')
    parser.add_argument('--source', type=str, default='FD001', help='源域数据集')
    parser.add_argument('--target', type=str, default='FD002', help='目标域数据集')
    parser.add_argument('--resume', action='store_true', help='从 checkpoint 续训')
    args = parser.parse_args()

    source_subset = args.source
    target_subset = args.target

    print("=" * 60)
    print(f"  🧪 跨工况迁移学习训练 —— TODO 5")
    print(f"  {source_subset} (源域) → {target_subset} (目标域)")
    print("=" * 60)

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  训练设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- 1. 加载数据 ----
    src_loader, tgt_loader, val_loader, src_edge, tgt_edge = \
        load_transfer_data(source_subset, target_subset)

    # ---- 2. 创建模型 ----
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
        fc_hidden=FC_HIDDEN_DIM
    ).to(device)

    # ---- 加载预训练权重（非续训时）----
    pretrain_path = f'saved_models/stgnn_best_{source_subset}.pt'
    if not args.resume:
        if os.path.exists(pretrain_path):
            pretrain = torch.load(pretrain_path, map_location=device)
            model.load_state_dict(pretrain['model_state_dict'])
            print(f"\n📥 已加载 {source_subset} 预训练权重 (Epoch {pretrain['epoch']+1})")
        else:
            print(f"\n⚠️  未找到预训练权重 {pretrain_path}，从头初始化")

    print(f"🔧 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 3. 损失函数和优化器 ----
    task_loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)

    print(f"  任务损失: CombinedLoss (MSE×{MSE_WEIGHT} + NASA×{NASA_SCORE_WEIGHT})")
    print(f"  LMMD 权重 λ: {LMMD_LAMBDA}")
    print(f"  优化器: Adam (lr={LEARNING_RATE * 0.1}, 微调学习率)")

    # ---- 4. 训练 ----
    checkpoint_path = f'saved_models/transfer_checkpoint.pt'
    model, train_losses, val_losses = train_transfer(
        model, src_loader, tgt_loader, val_loader,
        task_loss_fn, src_edge, tgt_edge, optimizer, device,
        target_subset=target_subset,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
        lmmd_lambda=LMMD_LAMBDA,
        resume=args.resume,
        checkpoint_path=checkpoint_path,
    )

    # ---- 5. 保存日志 ----
    save_training_log(train_losses, val_losses, source=source_subset, target=target_subset)

    print(f"\n🎉 TODO 5 动作1完成！迁移学习模型已保存。")
