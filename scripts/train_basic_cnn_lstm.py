# ============================================================
# scripts/train_basic_cnn_lstm.py —— CNN+LSTM 混合模型训练脚本
# ============================================================
# TODO 2.3 的核心脚本：在 FD001 数据集上训练 CNN+LSTM 混合模型
# 该模型不含图结构，用于证明显式建模传感器关联的必要性
#
# 功能:
#   1. 加载 data/processed 中预处理好的 FD001 数据
#   2. 使用 CNN_LSTM_Model（1D-CNN 提取局部特征 + LSTM 捕捉长时依赖）
#   3. 使用 CombinedLoss (MSE + NASA Score)
#   4. 支持训练中断后恢复（checkpoint 机制）
#   5. 早停（Early Stopping）防止过拟合
#   6. 训练完成后保存最佳模型到 saved_models/
#   7. 记录训练日志到 logs/
#
# ⚠️ 注意：所有超参数、数据加载方式、验证集划分方式
#   与 train_basic_lstm.py / train_basic_gru.py / train_basic_tcn.py 完全一致。
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

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
    RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT
)
from core_models.base_models import CNN_LSTM_Model
from utils.loss_functions import CombinedLoss
from utils.metrics import evaluate_metrics
from utils.data_processor import split_by_unit

# ============================================================
# 固定随机种子，保证可复现
# ============================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性训练（保证同种子可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 1. 加载预处理好的数据
# ============================================================
def load_data(subset='FD001', processed_dir='data/processed', val_ratio=0.2):
    """
    从 processed 目录加载预处理后的 .npz 数据，
    并从训练集中切出 80/20 作为 train/val（防止数据泄露）
    """
    train_path = os.path.join(processed_dir, f'{subset}_train.npz')
    test_path = os.path.join(processed_dir, f'{subset}_test.npz')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练数据: {train_path}，请先运行 data_processor.py 预处理数据")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试数据: {test_path}，请先运行 data_processor.py 预处理数据")

    # ---- 加载训练数据 ----
    train_data = np.load(train_path)
    X = train_data['X']  # [n_samples, WINDOW_SIZE, NUM_FEATURES]
    y = train_data['y']  # [n_samples]
    if 'unit' not in train_data.files:
        raise RuntimeError(
            f"❌ 数据缺少 unit 字段: {train_path}\n"
            f"   请重新运行数据预处理（python utils/data_processor.py）。"
        )
    unit_ids = train_data['unit']

    # ---- 加载测试数据 ----
    test_data = np.load(test_path)
    X_test_raw = test_data['X']  # [n_engines, WINDOW_SIZE, NUM_FEATURES]
    y_test_raw = test_data['y']  # [n_engines]

    print(f"\n📂 数据加载完成 - {subset}")
    print(f"  总训练样本: {len(X)}, 特征形状: {X.shape[1:]}")
    print(f"  测试引擎数: {len(X_test_raw)}")

    # ---- 拆分训练集 / 验证集 (80/20，按发动机分组防泄漏) ----
    X_train, X_val, y_train, y_val = split_by_unit(
        X, y, unit_ids, val_ratio=val_ratio, random_state=RANDOM_SEED
    )
    print(f"  训练样本: {len(X_train)}, 验证样本: {len(X_val)}")

    # ---- 转为 torch tensor ----
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test_raw, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1)

    # ---- 创建 DataLoader ----
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    print(f"📦 DataLoader: 训练批次 {len(train_loader)}, 验证批次 {len(val_loader)}")

    return train_loader, val_loader, (X_test, y_test)


# ============================================================
# 2. 保存 checkpoint（用于暂停后恢复训练）
# ============================================================
def save_checkpoint(model, optimizer, epoch, best_loss, train_losses, val_losses,
                    filepath='saved_models/cnn_lstm_checkpoint.pt'):
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
def load_checkpoint(model, optimizer, filepath='saved_models/cnn_lstm_checkpoint.pt'):
    """从 checkpoint 恢复训练状态"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"  🔄 从 Checkpoint 恢复训练，将从第 {start_epoch + 1} 轮继续")
        return start_epoch, best_loss, train_losses, val_losses
    else:
        print("  🆕 未找到 Checkpoint，从头开始训练")
        return 0, float('inf'), [], []


# ============================================================
# 4. 训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """执行一个训练 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播（CNN_LSTM_Model 内部自动处理 permute）
        y_pred = model(X_batch)

        # 计算损失
        loss = loss_fn(y_pred, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================
# 5. 验证
# ============================================================
def validate(model, dataloader, loss_fn, device):
    """在验证/测试集上评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            total_loss += loss.item()

            y_pred_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_batch.cpu().numpy())

    avg_loss = total_loss / num_batches
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    return avg_loss, y_pred_all, y_true_all


# ============================================================
# 6. 主训练函数
# ============================================================
def train(model, train_loader, val_loader, loss_fn, optimizer, device,
          num_epochs=NUM_EPOCHS, patience=EARLY_STOP_PATIENCE,
          resume=False, checkpoint_path='saved_models/cnn_lstm_checkpoint.pt'):
    """CNN+LSTM 模型主训练循环"""
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
    print(f"  🚀 开始训练 CNN+LSTM 混合模型")
    print(f"  设备: {device}, 最大轮数: {num_epochs}, 早停: {patience}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 训练阶段 ----
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # ---- 验证阶段 ----
        val_loss, y_pred, y_true = validate(model, val_loader, loss_fn, device)

        # ---- 记录 ----
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        # 每 5 轮打印一次详细信息
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

            best_model_path = 'saved_models/cnn_lstm_best_FD001.pt'
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

        # ---- 早停检查 ----
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
    log_path = f'logs/cnn_lstm_{subset}_{timestamp}.json'

    log_data = {
        'model': 'CNN_LSTM_Model',
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
# 8. 最终评估
# ============================================================
def final_evaluate(model, X_test, y_test, loss_fn, device, batch_size=BATCH_SIZE):
    """在测试集上做最终评估"""
    print(f"\n{'='*60}")
    print(f"  📊 最终测试集评估")
    print(f"{'='*60}")

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    val_loss, y_pred, y_true = validate(model, test_loader, loss_fn, device)

    evaluate_metrics(y_pred, y_true, print_result=True)

    return y_pred, y_true


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  🧪 CNN+LSTM 混合模型训练 —— TODO 2.3")
    print("=" * 60)

    # ---- 设备选择 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  训练设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- 1. 加载数据（训练集内 80/20 切分 train/val，与其他模型完全一致）----
    train_loader, val_loader, (X_test, y_test) = load_data(
        subset='FD001', val_ratio=0.2
    )

    # ---- 2. 创建 CNN+LSTM 混合模型 ----
    model = CNN_LSTM_Model(
        input_dim=NUM_FEATURES,   # 17（修正：使用全部特征，而非仅14个传感器）
        cnn_channels=64,          # CNN 输出通道数
        lstm_hidden=64,           # LSTM 隐藏层维度
        lstm_layers=2,            # LSTM 层数
        dropout=0.3
    ).to(device)

    print(f"\n🔧 模型: CNN_LSTM_Model")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 3. 定义损失函数和优化器（与其他模型完全一致）----
    loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"  损失函数: CombinedLoss (MSE×{MSE_WEIGHT} + NASA×{NASA_SCORE_WEIGHT})")
    print(f"  优化器: Adam (lr={LEARNING_RATE})")

    # ---- 4. 训练 ----
    try:
        model, train_losses, val_losses = train(
            model, train_loader, val_loader, loss_fn, optimizer, device,
            num_epochs=NUM_EPOCHS,
            patience=EARLY_STOP_PATIENCE,
            resume=False,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断 (Ctrl+C)")
        print("  当前模型已通过 checkpoint 自动保存，下次设置 resume=True 即可继续训练")
        sys.exit(0)

    # ---- 5. 保存训练日志 ----
    save_training_log(train_losses, val_losses, subset='FD001')

    # ---- 6. 加载最佳模型做最终评估 ----
    print(f"\n{'='*60}")
    print(f"  加载最佳模型进行最终评估...")
    print(f"{'='*60}")

    best_checkpoint = torch.load('saved_models/cnn_lstm_best_FD001.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"  已加载最佳模型 (Epoch {best_checkpoint['epoch']+1}, Val Loss: {best_checkpoint['best_loss']:.4f})")

    y_pred, y_true = final_evaluate(model, X_test, y_test, loss_fn, device)

    print(f"\n🎉 TODO 2.3 完成！CNN+LSTM 基线模型训练完毕。")
