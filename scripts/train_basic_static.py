# ============================================================
# scripts/train_basic_v2.py —— STGNN 主线模型训练脚本（v2：无 Transformer）
# ============================================================
# TODO 3 的核心脚本：在 FD001 数据集上训练 STGNN 模型
# 基于消融实验结果，采用无 Transformer 变体 (MSTCN + GAT)
#
# 功能:
#   1. 加载 data/processed 中预处理好的 FD001 数据
#   2. 加载图结构（edge_index）
#   3. 从训练集中拆分 80/20 作为 train/val（防止数据泄露）
#   4. 使用 STGNN 模型 + CombinedLoss (MSE + NASA Score)
#   5. 支持训练中断后恢复（checkpoint 机制）
#   6. 早停（Early Stopping）防止过拟合
#   7. 训练完成后保存最佳模型到 saved_models/
#   8. 记录训练日志到 logs/
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
from core_models.stgnn_static import STGNN_Static, repeat_edge_index_for_batch
from utils.loss_functions import CombinedLoss
from utils.metrics import evaluate_metrics

# ============================================================
# 固定随机种子，保证可复现
# ============================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1. 加载预处理好的数据 + 图结构
# ============================================================
def load_data_and_graph(subset='FD001', processed_dir='data/processed',
                        val_ratio=0.2):
    """
    从 processed 目录加载预处理后的数据和图结构，
    并从训练集中切出验证集

    参数:
        subset:         数据集编号（TODO 3 仅用 FD001）
        processed_dir: 预处理数据目录
        val_ratio:      验证集比例

    返回:
        train_loader, val_loader: 训练和验证 DataLoader
        edge_index:               图边索引
    """
    # ---- 加载训练数据 ----
    train_path = os.path.join(processed_dir, f'{subset}_train.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"找不到训练数据: {train_path}，请先运行 data_processor.py 预处理数据"
        )
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"找不到图结构: {graph_path}，请先运行 data_processor.py 预处理数据"
        )

    train_data = np.load(train_path)
    X = train_data['X']  # [n_samples, WINDOW_SIZE, NUM_FEATURES]
    y = train_data['y']  # [n_samples]

    graph = torch.load(graph_path)
    edge_index = graph['edge_index']  # [2, num_edges]

    print(f"\n📂 数据加载完成 - {subset}")
    print(f"  总样本数: {len(X)}, 特征形状: {X.shape[1:]}")
    print(f"  图边数: {edge_index.shape[1]}")

    # ---- 拆分训练集 / 验证集 ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=RANDOM_SEED, shuffle=True
    )
    print(f"  训练样本: {len(X_train)}, 验证样本: {len(X_val)}")

    # ---- 转为 torch tensor ----
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # ---- 创建 DataLoader ----
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, drop_last=False)

    print(f"📦 DataLoader: 训练批次 {len(train_loader)}, 验证批次 {len(val_loader)}")

    return train_loader, val_loader, edge_index


# ============================================================
# 2. 保存 checkpoint（暂停恢复）
# ============================================================
def save_checkpoint(model, optimizer, epoch, best_loss,
                    train_losses, val_losses,
                    filepath='saved_models/stgnn_v2_checkpoint.pt'):
    """保存训练状态，支持断点续训"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, filepath)
    print(f"  💾 Checkpoint 已保存 → {filepath}")


# ============================================================
# 3. 加载 checkpoint（恢复训练）
# ============================================================
def load_checkpoint(model, optimizer, filepath='saved_models/stgnn_v2_checkpoint.pt'):
    """从 checkpoint 恢复训练状态"""
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
# 4. 训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, loss_fn, optimizer, edge_index, device):
    """执行一个训练 epoch"""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        edge_index_device = edge_index.to(device)

        # 前向传播
        y_pred = model(X_batch, edge_index_device)

        # 计算损失
        loss = loss_fn(y_pred, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ============================================================
# 5. 验证
# ============================================================
def validate(model, dataloader, loss_fn, edge_index, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            edge_index_device = edge_index.to(device)

            y_pred = model(X_batch, edge_index_device)
            loss = loss_fn(y_pred, y_batch)

            total_loss += loss.item()
            y_pred_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    return avg_loss, y_pred_all, y_true_all


# ============================================================
# 6. 主训练函数
# ============================================================
def train(model, train_loader, val_loader, loss_fn, optimizer, edge_index, device,
          num_epochs=NUM_EPOCHS, patience=EARLY_STOP_PATIENCE,
          resume=False, checkpoint_path='saved_models/stgnn_v2_checkpoint.pt'):
    """STGNN (v2: 无 Transformer) 主训练循环"""
    # ---- 尝试恢复训练 ----
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
    print(f"  🚀 开始训练 STGNN 模型")
    print(f"  设备: {device}, 最大轮数: {num_epochs}, 早停: {patience}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 训练 ----
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, edge_index, device
        )

        # ---- 验证 ----
        val_loss, y_pred, y_true = validate(
            model, val_loader, loss_fn, edge_index, device
        )

        # ---- 记录 ----
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        # 每 5 轮打印详细信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            rmse, score = evaluate_metrics(y_pred, y_true, print_result=False)
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"RMSE: {rmse:.2f} | Score: {score:.1f} | "
                  f"耗时: {epoch_time:.1f}s")
        else:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"耗时: {epoch_time:.1f}s")

        # ---- 保存最佳模型 ----
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            best_model_path = 'saved_models/stgnn_static_best_FD001.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch,
            }, best_model_path)
            print(f"  ⭐ 新的最佳模型！Val Loss: {best_loss:.4f} → {best_model_path}")
        else:
            patience_counter += 1

        # ---- 保存 checkpoint ----
        save_checkpoint(model, optimizer, epoch, best_loss,
                        train_losses, val_losses, checkpoint_path)

        # ---- 早停 ----
        if patience_counter >= patience:
            print(f"\n  🛑 早停触发！验证 loss 连续 {patience} 轮未改善")
            break

    print(f"\n{'='*60}")
    print(f"  ✅ 训练完成！最佳验证损失: {best_loss:.4f}")
    print(f"{'='*60}")

    return model, train_losses, val_losses


# ============================================================
# 7. 保存训练日志
# ============================================================
def save_training_log(train_losses, val_losses, subset='FD001'):
    """将训练过程的损失记录保存为 JSON 文件"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/stgnn_static_{subset}_{timestamp}.json'

    log_data = {
        'model': 'STGNN_Static',
        'subset': subset,
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
    print("=" * 60)
    print("  🧪 STGNN 主线模型训练 —— TODO 3 (v2: 无 Transformer)")
    print("  MSTCN + GAT")
    print("=" * 60)

    # ---- 设备选择 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  训练设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- 1. 加载数据与图结构 ----
    train_loader, val_loader, edge_index = load_data_and_graph(
        subset='FD001', val_ratio=0.2
    )

    # ---- 2. 创建模型 ----
    model = STGNN_Static(
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

    print(f"\n🔧 模型: STGNN (MSTCN + GAT, 无 Transformer)")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 3. 定义损失函数和优化器 ----
    loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"  损失函数: CombinedLoss (MSE×{MSE_WEIGHT} + NASA×{NASA_SCORE_WEIGHT})")
    print(f"  优化器: Adam (lr={LEARNING_RATE})")

    # ---- 4. 训练 ----
    model, train_losses, val_losses = train(
        model, train_loader, val_loader, loss_fn, optimizer, edge_index, device,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
        resume=False,  # ← 改为 True 可恢复训练
    )

    # ---- 5. 保存训练日志 ----
    save_training_log(train_losses, val_losses, subset='FD001')

    # ---- 6. 加载最佳模型 ----
    print(f"\n{'='*60}")
    print(f"  加载最佳模型...")
    print(f"{'='*60}")

    best_checkpoint = torch.load('saved_models/stgnn_static_best_FD001.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"  已加载最佳模型 (Epoch {best_checkpoint['epoch']+1}, "
          f"Val Loss: {best_checkpoint['best_loss']:.4f})")

    # ---- 7. 验证集最终评估 ----
    val_loss, y_pred, y_true = validate(
        model, val_loader, loss_fn, edge_index, device
    )
    print(f"\n📊 验证集最终评估:")
    evaluate_metrics(y_pred, y_true, print_result=True)

    print(f"\n🎉 TODO 3 完成！STGNN v2 (MSTCN + GAT) 主线模型训练完毕。")
    print(f"  最佳模型保存在: saved_models/stgnn_static_best_FD001.pt")
