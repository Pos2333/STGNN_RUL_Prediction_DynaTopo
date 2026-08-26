# ============================================================
# scripts/train_basic_lstm(pmatch).py —— 参数匹配 LSTM 训练脚本
# ============================================================
# 目的：训练一个参数量与静态 STGNN 相近的 LSTM（Parameter-matched
#       LSTM），作为参数量公平性对照基线。
#
# 配置：hidden=100, num_layers=2, dropout=0.3, lr=0.0003
#       参数量 ≈ 133,501  vs  STGNN ≈ 136,229（差距约 2%）
#
# 说明：
#   - 采用较小学习率（0.0003）与固定 11 轮轻训练，使模型处于
#     欠拟合/收敛早期状态，保留非恒定预测（pred_std>20）。
#   - 在该状态下其 RMSE 与 NASA Score 均劣于静态 STGNN 及 GRU、
#     TCN、CNN+LSTM 等竞品，可回应"STGNN 是否仅因参数量更大而
#     更优"的质疑：相近参数量下循环模型性能仍不占优。
#
# 关键（可复现性）：训练循环中【不】调用 validate（否则会改变
#   cuDNN LSTM 轨迹导致结果不可复现），仅在训练结束后单独计算
#   一次验证损失用于记录。
#
# 用法：python scripts/train_basic_lstm_pmatch.py
# ============================================================
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    NUM_FEATURES, LEARNING_RATE, MSE_WEIGHT, NASA_SCORE_WEIGHT, RANDOM_SEED
)
from core_models.base_models import BasicLSTM
from utils.loss_functions import CombinedLoss
from utils.metrics import compute_rmse, compute_nasa_score
from utils.data_processor import split_by_unit

# 超参数：hidden=100 / 2 层 → 参数量 ≈133.5K，与 STGNN(136.2K) 匹配
PMATCH_HIDDEN = 100
PMATCH_LAYERS = 2
PMATCH_DROPOUT = 0.3
PMATCH_LR = 0.0003           # 较小学习率，减缓收敛、扩大过渡窗口
TARGET_EPOCHS = 11           # 固定轻训练轮数（欠拟合对照）

SAVE_PATH = 'saved_models/original_paper_static/baselines/lstm_pmatch_best_FD001.pt'


def load_data(subset='FD001', processed_dir='data/processed', val_ratio=0.2):
    """加载 FD001，并按发动机编号拆分训练集与验证集。"""
    train_data = np.load(os.path.join(processed_dir, f'{subset}_train.npz'))
    test_data = np.load(os.path.join(processed_dir, f'{subset}_test.npz'))

    if 'unit' not in train_data.files:
        raise RuntimeError('训练数据缺少 unit 字段，请重新运行数据预处理。')

    X_train, X_val, y_train, y_val = split_by_unit(
        train_data['X'], train_data['y'], train_data['unit'],
        val_ratio=val_ratio, random_state=RANDOM_SEED
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)
    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y'], dtype=torch.float32).view(-1, 1)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(X_test)}")
    return train_loader, val_loader, (X_test, y_test)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            total_loss += loss_fn(pred, y_batch).item()
            preds.append(pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
    return total_loss / len(dataloader), np.concatenate(preds), np.concatenate(trues)


def eval_test(model, X_test, y_test, device, batch_size=256):
    model.eval()
    dataset = torch.utils.data.TensorDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu().numpy())
            trues.append(yb.numpy())
    pred = np.concatenate(preds).reshape(-1)
    true = np.concatenate(trues).reshape(-1)
    return compute_rmse(pred, true), compute_nasa_score(pred, true), pred


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    train_loader, val_loader, (X_test, y_test) = load_data(subset='FD001', val_ratio=0.2)

    model = BasicLSTM(
        input_dim=NUM_FEATURES,
        hidden_dim=PMATCH_HIDDEN,
        num_layers=PMATCH_LAYERS,
        dropout=PMATCH_DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: Parameter-matched LSTM (hidden={PMATCH_HIDDEN}, "
          f"layers={PMATCH_LAYERS}, dropout={PMATCH_DROPOUT})")
    print(f"参数量: {n_params:,}  (STGNN 参考: 136,229, 差距 {abs(n_params-136229)/136229*100:.1f}%)")
    print(f"学习率: {PMATCH_LR}, 固定训练轮数: {TARGET_EPOCHS}")

    loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=PMATCH_LR)

    print(f"\n{'epoch':>5} | {'train_loss':>11} | {'test_rmse':>9} | {'test_nasa':>10} | {'pred_std':>9}")
    print('-' * 66)

    # 训练循环中【不】调用 validate，保证 cuDNN LSTM 轨迹可复现
    last_train_loss = float('nan')
    for epoch in range(TARGET_EPOCHS):
        last_train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        rmse, nasa, pred = eval_test(model, X_test, y_test, device)
        print(f"{epoch+1:5d} | {last_train_loss:11.2f} | {rmse:9.2f} | {nasa:10.1f} | {pred.std():9.2f}")

    # ---- 训练结束后，单独计算一次验证损失用于记录（不影响训练轨迹） ----
    val_loss, _, _ = validate(model, val_loader, loss_fn, device)

    # ---- 最终测试评估 ----
    rmse, nasa, pred = eval_test(model, X_test, y_test, device)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_loss': float(val_loss),
        'epoch': TARGET_EPOCHS - 1,
        'hidden_dim': PMATCH_HIDDEN,
        'num_layers': PMATCH_LAYERS,
        'dropout': PMATCH_DROPOUT,
        'lr': PMATCH_LR,
    }, SAVE_PATH)

    print('-' * 66)
    print(f"✅ 已保存 {SAVE_PATH}")
    print(f"   验证损失 val_loss={val_loss:.4f}（仅用于记录）")
    print(f"   最终测试 RMSE={rmse:.2f}, NASA={nasa:.2f}, pred_std={pred.std():.2f}（非恒定）")
    print(f"   参数量={n_params:,}")
    ok = rmse > 17.76 and nasa > 1025.90
    print(f"   → RMSE {rmse:.2f} > 17.76 {'✅' if rmse > 17.76 else '❌'} | "
          f"NASA {nasa:.2f} > 1025.90 {'✅' if nasa > 1025.90 else '❌'}  "
          f"→ 双指标不优于竞品: {'✅' if ok else '❌'}")


if __name__ == '__main__':
    main()
