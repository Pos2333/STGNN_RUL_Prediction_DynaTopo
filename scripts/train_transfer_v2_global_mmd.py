# ============================================================
# scripts/train_transfer_v2_global_mmd.py
# 全局 MMD 迁移训练（与 LMMD 对比用）
# ============================================================
# 基于 FD001 预训练模型，用全局 MMD 迁移到 FD002
#
# 与 UDA LMMD 的区别:
#   - LMMD: 分子域对齐 (lmmd_loss)
#   - 全局 MMD: 不分子域，整体拉近 (mmd_loss)
#
# 用法: python scripts/train_transfer_v2_global_mmd.py --target FD002
# ============================================================
import os, sys, json, time, datetime, argparse
import numpy as np
import torch, torch.nn as nn
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
from utils.loss_functions import CombinedLoss, mmd_loss
from utils.metrics import evaluate_metrics

torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='FD002')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--mmd_lambda', type=float, default=0.1)
args = parser.parse_args()

SOURCE = 'FD001'
TARGET = args.target
MMD_LAMBDA = args.mmd_lambda
PROCESSED_DIR = 'data/processed'

print(f"{'='*60}")
print(f"  全局 MMD 迁移训练: {SOURCE} → {TARGET}")
print(f"  MMD λ = {MMD_LAMBDA}")
print(f"  设备: {DEVICE}")
print(f"{'='*60}")

# ============================================================
# 1. 加载数据
# ============================================================
src_data = np.load(os.path.join(PROCESSED_DIR, f'{SOURCE}_train.npz'))
X_src, y_src = src_data['X'], src_data['y']
src_edge = torch.load(os.path.join(PROCESSED_DIR, f'{SOURCE}_train_graph.pt'), weights_only=False)['edge_index']

tgt_data = np.load(os.path.join(PROCESSED_DIR, f'{TARGET}_train.npz'))
X_tgt, y_tgt = tgt_data['X'], tgt_data['y']

tgt_graph_path = os.path.join(PROCESSED_DIR, f'{TARGET}_train_graph.pt')
tgt_edge = torch.load(tgt_graph_path, weights_only=False)['edge_index'] if os.path.exists(tgt_graph_path) else src_edge

print(f"源域 {SOURCE}: {len(X_src)} 样本, 边={src_edge.shape[1]}")
print(f"目标域 {TARGET}: {len(X_tgt)} 样本, 边={tgt_edge.shape[1]}")

# 切分
X_src_tr, X_src_val, y_src_tr, y_src_val = train_test_split(X_src, y_src, test_size=0.15, random_state=RANDOM_SEED)
X_tgt_tr, X_tgt_val, y_tgt_tr, y_tgt_val = train_test_split(X_tgt, y_tgt, test_size=0.15, random_state=RANDOM_SEED)

def to_loader(X, y, shuffle=True):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32).view(-1, 1)),
                      batch_size=BATCH_SIZE, shuffle=shuffle)

src_train_ld = to_loader(X_src_tr, y_src_tr, True)
src_val_ld   = to_loader(X_src_val, y_src_val, False)
tgt_train_ld = to_loader(X_tgt_tr, y_tgt_tr, True)
tgt_val_ld   = to_loader(X_tgt_val, y_tgt_val, False)

# ============================================================
# 2. 构建模型 & 加载预训练权重
# ============================================================
model = STGNN(
    num_sensors=14, num_op_settings=3,
    mstcn_channels=MSTCN_NUM_CHANNELS, mstcn_kernels=MSTCN_KERNEL_SIZES, mstcn_dropout=MSTCN_DROPOUT,
    gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_HEADS, gat_dropout=GAT_DROPOUT,
    trans_d_model=TRANSFORMER_D_MODEL, trans_nhead=TRANSFORMER_NHEAD, trans_num_layers=TRANSFORMER_NUM_LAYERS,
    trans_dropout=TRANSFORMER_DROPOUT, use_mstcn=True, use_gat=True, use_transformer=False, fc_hidden=FC_HIDDEN_DIM,
).to(DEVICE)

pretrained_path = 'saved_models/stgnn_v2_best_FD001.pt'
ck = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ck['model_state_dict'])
print(f"已加载预训练模型: {pretrained_path}")

task_loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ============================================================
# 3. Checkpoint 逻辑
# ============================================================
CKPT_PATH = f'saved_models/transfer_v2_global_mmd_checkpoint_{TARGET}.pt'
BEST_PATH  = f'saved_models/transfer_v2_global_mmd_best_{TARGET}.pt'
start_epoch, best_tgt_val_loss, patience_counter = 0, float('inf'), 0
train_losses, tgt_val_losses = [], []

if args.resume and os.path.exists(CKPT_PATH):
    cp = torch.load(CKPT_PATH, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    start_epoch = cp['epoch'] + 1
    best_tgt_val_loss = cp.get('best_tgt_val_loss', float('inf'))
    train_losses = cp.get('train_losses', [])
    tgt_val_losses = cp.get('tgt_val_losses', [])
    print(f"从 Checkpoint 恢复: epoch {start_epoch}")

# ============================================================
# 4. 训练循环
# ============================================================
print(f"\n{'='*60}")
print(f"  开始全局 MMD 训练 ({SOURCE} → {TARGET})")
print(f"{'='*60}")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_task, total_mmd, total_loss_sum = 0.0, 0.0, 0.0
    src_iter = iter(src_train_ld); tgt_iter = iter(tgt_train_ld)
    n_batches = max(len(src_train_ld), len(tgt_train_ld))

    for _ in range(n_batches):
        try: X_s, y_s = next(src_iter)
        except StopIteration: src_iter = iter(src_train_ld); X_s, y_s = next(src_iter)
        try: X_t, y_t = next(tgt_iter)
        except StopIteration: tgt_iter = iter(tgt_train_ld); X_t, y_t = next(tgt_iter)

        X_s, y_s = X_s.to(DEVICE), y_s.to(DEVICE)
        X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
        src_e, tgt_e = src_edge.to(DEVICE), tgt_edge.to(DEVICE)

        pred_s, feat_s = model(X_s, src_e, return_feat=True)
        pred_t, feat_t = model(X_t, tgt_e, return_feat=True)

        task_loss = task_loss_fn(pred_s, y_s) + TGT_TASK_WEIGHT * task_loss_fn(pred_t, y_t)
        mmd_val = mmd_loss(feat_s, feat_t)  # 全局 MMD，无子域划分
        total_loss = task_loss + MMD_LAMBDA * mmd_val

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_task += task_loss.item(); total_mmd += mmd_val.item(); total_loss_sum += total_loss.item()

    avg_task = total_task / n_batches; avg_mmd = total_mmd / n_batches

    # 目标域验证
    model.eval()
    tgt_val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in tgt_val_ld:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            tgt_val_loss += task_loss_fn(model(X_b, tgt_edge.to(DEVICE)), y_b).item()
    tgt_val_loss /= len(tgt_val_ld)

    train_losses.append(total_loss_sum / n_batches); tgt_val_losses.append(tgt_val_loss)
    scheduler.step(tgt_val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Task={avg_task:.2f} MMD={avg_mmd:.4f} TgtVal={tgt_val_loss:.2f}")

    # 早停 & 保存
    if tgt_val_loss < best_tgt_val_loss - 1e-4:
        best_tgt_val_loss = tgt_val_loss; patience_counter = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_tgt_val_loss': best_tgt_val_loss}, BEST_PATH)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"早停于 epoch {epoch+1}")
            break

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_tgt_val_loss': best_tgt_val_loss,
                'train_losses': train_losses, 'tgt_val_losses': tgt_val_losses}, CKPT_PATH)

# ============================================================
# 5. 最终评估
# ============================================================
print(f"\n加载最佳模型进行评估...")
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)['model_state_dict'])
model.eval()

def full_predict(loader, edge):
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b.to(DEVICE), edge.to(DEVICE)).cpu().numpy())
            trues.append(y_b.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)

# 目标域测试集
tgt_test = np.load(os.path.join(PROCESSED_DIR, f'{TARGET}_test.npz'))
X_tt, y_tt = tgt_test['X'], tgt_test['y']
tgt_test_ld = DataLoader(TensorDataset(torch.tensor(X_tt, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)

tgt_preds = []
with torch.no_grad():
    for (X_b,) in tgt_test_ld:
        tgt_preds.append(model(X_b.to(DEVICE), tgt_edge.to(DEVICE)).cpu().numpy())
tgt_preds = np.concatenate(tgt_preds)

tgt_rmse, tgt_score = evaluate_metrics(tgt_preds, y_tt, print_result=True)

# 保存日志
log = {
    'model': 'STGNN_GlobalMMD_v2', 'source': SOURCE, 'target': TARGET,
    'strategy': 'GlobalMMD', 'mmd_lambda': MMD_LAMBDA,
    'tgt_rmse': float(tgt_rmse), 'tgt_score': float(tgt_score),
    'best_tgt_val_loss': float(best_tgt_val_loss), 'num_epochs': epoch + 1,
    'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
}
log_path = f'logs/transfer_v2_global_mmd_{SOURCE}_to_{TARGET}_{log["timestamp"]}.json'
with open(log_path, 'w') as f: json.dump(log, f, indent=2)
print(f"日志已保存: {log_path}")
print(f"\n✅ 全局 MMD 训练完成!")
