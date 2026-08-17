# ============================================================
# scripts/train_transfer.py —— 跨工况迁移学习训练脚本（统一版）
# ============================================================
# 支持 static / A1B1 / A1B2 等模型的半监督 LMMD 跨工况迁移。
#
# 策略：
#   1. 加载 FD001 上训练好的模型作为预训练起点（static 或 dynatopo）
#   2. 同时加载 FD001（源域）和 目标域 数据
#   3. 源域和目标域分别计算 CombinedLoss（MSE + NASA Score）
#   4. 对源域和目标域的中间特征计算 LMMD 损失（双向子域对齐）
#   5. 总损失 = 源域任务损失 + w * 目标域任务损失 + λ * LMMD
#   6. 同时在源域和目标域验证集上监控性能，早停基于目标域验证损失
#   7. 支持 checkpoint 暂停恢复
#
# 用法：
#   python scripts/train_transfer.py --preset static --target FD002   # 静态基线迁移
#   python scripts/train_transfer.py --preset A1B1 --target FD002     # A1B1 迁移
#   python scripts/train_transfer.py --preset A1B2 --target all       # A1B2 全目标域
#   python scripts/train_transfer.py --preset A1B1 --target FD002 --resume  # 续训
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
from core_models.stgnn_static import STGNN_Static
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from configs.dynatopo_config import get_experiment_config
from utils.loss_functions import CombinedLoss, lmmd_loss, mmd_loss
from utils.metrics import evaluate_metrics
from utils.data_processor import split_by_unit

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性训练（保证同种子可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 1. 加载源域和目标域数据（目标域也拆出验证集）
# ============================================================
def load_transfer_data(source_subset='FD001', target_subset='FD002',
                       processed_dir='data/processed', val_ratio=0.15):
    """
    加载源域和目标域的训练数据，两者都拆出验证集

    返回:
        src_train_loader, src_val_loader: 源域训练/验证 DataLoader
        tgt_train_loader, tgt_val_loader: 目标域训练/验证 DataLoader
        src_edge, tgt_edge:               图边索引
    """
    # ---- 加载源域数据（FD001） ----
    src_path = os.path.join(processed_dir, f'{source_subset}_train.npz')
    src_graph_path = os.path.join(processed_dir, f'{source_subset}_train_graph.pt')
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"找不到源域数据: {src_path}")

    src_data = np.load(src_path)
    X_src = src_data['X']
    y_src = src_data['y']
    if 'unit' not in src_data.files:
        raise RuntimeError(
            f"❌ 源域数据缺少 unit 字段: {src_path}\n"
            f"   请重新运行数据预处理（python utils/data_processor.py）。"
        )
    unit_src = src_data['unit']
    src_graph = torch.load(src_graph_path, weights_only=False)
    src_edge = src_graph['edge_index']

    print(f"\n📂 源域 {source_subset}: {len(X_src)} 个训练样本")
    print(f"  源域图边数: {src_edge.shape[1]}")

    # ---- 加载目标域数据 ----
    tgt_path = os.path.join(processed_dir, f'{target_subset}_train.npz')
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"找不到目标域数据: {tgt_path}")

    tgt_data = np.load(tgt_path)
    X_tgt = tgt_data['X']
    y_tgt = tgt_data['y']
    if 'unit' not in tgt_data.files:
        raise RuntimeError(
            f"❌ 目标域数据缺少 unit 字段: {tgt_path}\n"
            f"   请重新运行数据预处理（python utils/data_processor.py）。"
        )
    unit_tgt = tgt_data['unit']

    print(f"📂 目标域 {target_subset}: {len(X_tgt)} 个训练样本")

    # ---- 目标域图结构 ----
    tgt_graph_path = os.path.join(processed_dir, f'{target_subset}_train_graph.pt')
    if os.path.exists(tgt_graph_path):
        tgt_graph = torch.load(tgt_graph_path, weights_only=False)
        tgt_edge = tgt_graph['edge_index']
        print(f"  目标域图边数: {tgt_edge.shape[1]}")
    else:
        tgt_edge = src_edge
        print(f"  目标域使用源域图结构（{tgt_edge.shape[1]} 边）")

    # ---- 源域拆出验证集（按发动机分组，防泄漏）----
    X_src_train, X_src_val, y_src_train, y_src_val = split_by_unit(
        X_src, y_src, unit_src, val_ratio=val_ratio, random_state=RANDOM_SEED
    )

    # ---- 目标域也拆出验证集（监控目标域泛化性能，同样按发动机分组）----
    X_tgt_train, X_tgt_val, y_tgt_train, y_tgt_val = split_by_unit(
        X_tgt, y_tgt, unit_tgt, val_ratio=val_ratio, random_state=RANDOM_SEED
    )

    print(f"  源域: 训练 {len(X_src_train)}, 验证 {len(X_src_val)}")
    print(f"  目标域: 训练 {len(X_tgt_train)}, 验证 {len(X_tgt_val)}")

    # ---- 转为 tensor ----
    def to_tensor(X, y):
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).view(-1, 1))

    X_src_train_t, y_src_train_t = to_tensor(X_src_train, y_src_train)
    X_src_val_t, y_src_val_t = to_tensor(X_src_val, y_src_val)
    X_tgt_train_t, y_tgt_train_t = to_tensor(X_tgt_train, y_tgt_train)
    X_tgt_val_t, y_tgt_val_t = to_tensor(X_tgt_val, y_tgt_val)

    # ---- DataLoader ----
    def make_loader(X, y, shuffle=True):
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE,
                          shuffle=shuffle, drop_last=False)

    src_train_loader = make_loader(X_src_train_t, y_src_train_t, shuffle=True)
    src_val_loader = make_loader(X_src_val_t, y_src_val_t, shuffle=False)
    tgt_train_loader = make_loader(X_tgt_train_t, y_tgt_train_t, shuffle=True)
    tgt_val_loader = make_loader(X_tgt_val_t, y_tgt_val_t, shuffle=False)

    print(f"📦 源域批次: {len(src_train_loader)}, 目标域批次: {len(tgt_train_loader)}")

    return (src_train_loader, src_val_loader,
            tgt_train_loader, tgt_val_loader,
            src_edge, tgt_edge)


# ============================================================
# 2. 保存 / 加载 checkpoint
# ============================================================
def save_checkpoint(state, filepath):
    torch.save(state, filepath)
    print(f"  💾 Checkpoint 已保存 → {filepath}")


def load_checkpoint(filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        print(f"  🔄 从 Checkpoint 恢复，从第 {checkpoint['epoch'] + 2} 轮继续")
        return checkpoint
    else:
        print("  🆕 未找到 Checkpoint，从头开始训练")
        return None


# ============================================================
# 3. 训练一个 epoch（含 LMMD 迁移损失 + 目标域监督 + 双向子域对齐）
# ============================================================
def train_one_epoch_transfer(model, src_loader, tgt_loader,
                             task_loss_fn, src_edge, tgt_edge, optimizer,
                             device, lmmd_lambda=LMMD_LAMBDA,
                             tgt_task_weight=TGT_TASK_WEIGHT,
                             adapt_mode='lmmd_semi'):
    """
    迁移学习训练一个 epoch

    adapt_mode 说明：
      - lmmd_semi:  源域监督 + 目标域监督 + 双向 LMMD（默认）
      - lmmd_uda:   源域监督 + 单向 LMMD（无目标域标签监督）
      - global_mmd: 源域监督 + 目标域监督 + 全局 MMD
      - none:       源域监督 + 目标域监督（无域自适应损失）
    """
    model.train()
    total_task_src = 0.0
    total_task_tgt = 0.0
    total_lmmd = 0.0
    total_loss_sum = 0.0

    num_batches = max(len(src_loader), len(tgt_loader))
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    for _ in range(num_batches):
        # ---- 取 batch ----
        try:
            X_src, y_src = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            X_src, y_src = next(src_iter)

        try:
            X_tgt, y_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            X_tgt, y_tgt = next(tgt_iter)

        X_src, y_src = X_src.to(device), y_src.to(device)
        X_tgt, y_tgt = X_tgt.to(device), y_tgt.to(device)
        src_edge_d = src_edge.to(device)
        tgt_edge_d = tgt_edge.to(device)

        # ---- 前向传播（获取预测值和融合特征） ----
        y_pred_src, feat_src = model(X_src, src_edge_d, return_feat=True)
        y_pred_tgt, feat_tgt = model(X_tgt, tgt_edge_d, return_feat=True)

        # ---- 源域任务损失（所有模式都有） ----
        loss_src = task_loss_fn(y_pred_src, y_src)

        # ---- 目标域任务损失（UDA 无监督模式没有） ----
        if adapt_mode == 'lmmd_uda':
            loss_tgt = torch.tensor(0.0, device=device)
            task_loss = loss_src
        else:
            loss_tgt = task_loss_fn(y_pred_tgt, y_tgt)
            task_loss = loss_src + tgt_task_weight * loss_tgt

        # ---- 域自适应损失 ----
        if adapt_mode == 'none':
            lmmd_val = torch.tensor(0.0, device=device)
        elif adapt_mode == 'global_mmd':
            lmmd_val = mmd_loss(feat_src, feat_tgt)
        elif adapt_mode == 'lmmd_uda':
            # 无监督：仅源域标签划分（目标域无标签，不做双向）
            lmmd_val = lmmd_loss(feat_src, feat_tgt, y_src)
        else:  # lmmd_semi
            lmmd_s2t = lmmd_loss(feat_src, feat_tgt, y_src)
            lmmd_t2s = lmmd_loss(feat_tgt, feat_src, y_tgt)
            lmmd_val = (lmmd_s2t + lmmd_t2s) / 2.0

        # ---- 总损失 ----
        total_loss = task_loss + lmmd_lambda * lmmd_val

        # ---- 反向传播 ----
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_task_src += loss_src.item()
        total_task_tgt += loss_tgt.item()
        total_lmmd += lmmd_val.item()
        total_loss_sum += total_loss.item()

    n = num_batches
    return total_task_src / n, total_task_tgt / n, total_lmmd / n, total_loss_sum / n


# ============================================================
# 4. 验证（在给定 DataLoader 上）
# ============================================================
@torch.no_grad()
def evaluate_on_loader(model, loader, task_loss_fn, edge_index, device):
    """在任意 DataLoader 上评估，返回 loss, y_pred, y_true"""
    model.eval()
    total_loss = 0.0
    y_pred_all, y_true_all = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        edge_d = edge_index.to(device)

        y_pred = model(X_batch, edge_d)
        loss = task_loss_fn(y_pred, y_batch)

        total_loss += loss.item()
        y_pred_all.append(y_pred.cpu().numpy())
        y_true_all.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    return avg_loss, y_pred_all, y_true_all


# ============================================================
# 5. 主训练函数
# ============================================================
def train_transfer(model, src_train_loader, src_val_loader,
                   tgt_train_loader, tgt_val_loader,
                   task_loss_fn, src_edge, tgt_edge, optimizer, device,
                   target_subset='FD002',
                   num_epochs=NUM_EPOCHS, patience=EARLY_STOP_PATIENCE,
                   lmmd_lambda=LMMD_LAMBDA,
                   resume=False,
                   checkpoint_path='saved_models/original_paper_static/transfer/lmmd_semi/transfer_static_lmmd_semi_checkpoint_FD002.pt',
                   prefix='static',
                   adapt_mode='lmmd_semi'):
    """
    迁移学习主训练循环

    改进：
      - 同时在源域和目标域验证集上监控
      - 早停基于目标域验证损失（更关注迁移效果）
      - 记录源域和目标域的 RMSE/Score
    """
    if resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_tgt_val_loss = ckpt.get('best_tgt_val_loss', float('inf'))
            train_losses = ckpt.get('train_losses', [])
            tgt_val_losses = ckpt.get('tgt_val_losses', [])
        else:
            start_epoch = 0
            best_tgt_val_loss = float('inf')
            train_losses = []
            tgt_val_losses = []
    else:
        start_epoch = 0
        best_tgt_val_loss = float('inf')
        train_losses = []
        tgt_val_losses = []

    patience_counter = 0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"  🚀 开始迁移学习训练 ({target_subset}, mode={adapt_mode})")
    print(f"  设备: {device}, 最大轮数: {num_epochs}, 早停: {patience}")
    print(f"  LMMD λ: {lmmd_lambda}, 目标域任务权重 w: {TGT_TASK_WEIGHT}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 训练 ----
        task_src, task_tgt, lmmd_val, total_loss = train_one_epoch_transfer(
            model, src_train_loader, tgt_train_loader,
            task_loss_fn, src_edge, tgt_edge, optimizer, device,
            lmmd_lambda, TGT_TASK_WEIGHT, adapt_mode
        )

        # ---- 源域验证 ----
        src_val_loss, src_pred, src_true = evaluate_on_loader(
            model, src_val_loader, task_loss_fn, src_edge, device
        )

        # ---- 目标域验证（关键！监控迁移效果） ----
        tgt_val_loss, tgt_pred, tgt_true = evaluate_on_loader(
            model, tgt_val_loader, task_loss_fn, tgt_edge, device
        )

        train_losses.append(total_loss)
        tgt_val_losses.append(tgt_val_loss)

        epoch_time = time.time() - epoch_start

        # ---- 打印 ----
        if (epoch + 1) % 5 == 0 or epoch == 0:
            src_rmse, src_score = evaluate_metrics(src_pred, src_true, print_result=False)
            tgt_rmse, tgt_score = evaluate_metrics(tgt_pred, tgt_true, print_result=False)
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Src: {task_src:.2f} | Tgt: {task_tgt:.2f} | "
                  f"LMMD: {lmmd_val:.4f} | "
                  f"Val(S): {src_val_loss:.2f} | Val(T): {tgt_val_loss:.2f} | "
                  f"⏱ {epoch_time:.1f}s")
            print(f"          📊 源域: RMSE={src_rmse:.2f}, Score={src_score:.1f} | "
                  f"目标域: RMSE={tgt_rmse:.2f}, Score={tgt_score:.1f}")
        else:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Src: {task_src:.2f} | Tgt: {task_tgt:.2f} | "
                  f"LMMD: {lmmd_val:.4f} | "
                  f"Val(S): {src_val_loss:.2f} | Val(T): {tgt_val_loss:.2f} | "
                  f"⏱ {epoch_time:.1f}s")

        # ---- 保存最佳模型（基于目标域验证损失） ----
        if tgt_val_loss < best_tgt_val_loss:
            best_tgt_val_loss = tgt_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if prefix == 'static':
                best_model_path = (
                    f'saved_models/original_paper_static/transfer/{adapt_mode}/'
                    f'transfer_static_{adapt_mode}_best_{target_subset}.pt'
                )
            else:
                best_model_path = f'saved_models/transfer_{prefix}_{adapt_mode}_best_{target_subset}.pt'
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_tgt_val_loss': best_tgt_val_loss,
                'epoch': epoch,
            }, best_model_path)
            print(f"  ⭐ 新的最佳模型！（目标域 Val Loss: {best_tgt_val_loss:.4f}）")
        else:
            patience_counter += 1

        # ---- 保存 checkpoint ----
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_tgt_val_loss': best_tgt_val_loss,
            'train_losses': train_losses,
            'tgt_val_losses': tgt_val_losses,
        }, checkpoint_path)

        # ---- 早停 ----
        if patience_counter >= patience:
            print(f"\n  🛑 早停触发！目标域验证 loss 连续 {patience} 轮未改善")
            break

    # ---- 恢复最佳模型 ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n{'='*60}")
    print(f"  ✅ 迁移学习训练完成！最佳目标域验证损失: {best_tgt_val_loss:.4f}")
    print(f"{'='*60}")

    return model, train_losses, tgt_val_losses


# ============================================================
# 6. 保存训练日志
# ============================================================
def save_training_log(train_losses, val_losses, source='FD001', target='FD002',
                      prefix='static', adapt_mode='lmmd_semi'):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/transfer_{prefix}_{adapt_mode}_{source}_to_{target}_{timestamp}.json'

    log_data = {
        'model': f'STGNN_Transfer_{prefix}',
        'source': source,
        'target': target,
        'adapt_mode': adapt_mode,
        'lmmd_lambda': LMMD_LAMBDA,
        'tgt_task_weight': TGT_TASK_WEIGHT,
        'train_losses': train_losses,
        'tgt_val_losses': val_losses,
        'best_tgt_val_loss': min(val_losses) if val_losses else None,
        'num_epochs': len(train_losses),
    }

    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"\n📝 训练日志已保存 → {log_path}")


# ============================================================
# 7. 构建模型并加载预训练权重
# ============================================================
def build_and_load_model(source_subset, device, resume, pretrain_path, preset='static'):
    """
    构建模型（static 或 dynatopo），加载 FD001 预训练权重

    改进：预训练模型不存在时，自动从 ablation 实验模型复制；
          两者都不存在则明确报错，而非静默随机初始化
    """
    if preset == 'static':
        model = STGNN_Static(
            num_sensors=14, num_op_settings=3,
            mstcn_channels=MSTCN_NUM_CHANNELS, mstcn_kernels=MSTCN_KERNEL_SIZES,
            mstcn_dropout=MSTCN_DROPOUT,
            gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_HEADS, gat_dropout=GAT_DROPOUT,
            trans_d_model=TRANSFORMER_D_MODEL, trans_nhead=TRANSFORMER_NHEAD,
            trans_num_layers=TRANSFORMER_NUM_LAYERS, trans_dropout=TRANSFORMER_DROPOUT,
            use_transformer=False,
            fc_hidden=FC_HIDDEN_DIM
        )
    else:
        cfg = get_experiment_config(preset)
        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3, fc_hidden=FC_HIDDEN_DIM)
    model = model.to(device)

    if not resume:
        if not os.path.exists(pretrain_path):
            # 尝试从 ablation 模型复制
            alt_path = 'saved_models/original_paper_static/ablation/ablation_无_Transformer.pt'
            if os.path.exists(alt_path):
                print(f"\n⚠️  未找到 {pretrain_path}")
                print(f"  💡 找到消融实验模型 {alt_path}，将复制作为预训练权重")
                alt_state = torch.load(alt_path, map_location=device, weights_only=False)
                # ablation 模型保存的是裸 state_dict
                if 'model_state_dict' in alt_state:
                    sd = alt_state['model_state_dict']
                else:
                    sd = alt_state
                # 保存为标准 checkpoint 格式
                torch.save({'model_state_dict': sd, 'epoch': 0, 'best_loss': 0},
                           pretrain_path)
                print(f"  ✅ 已创建 {pretrain_path}")
            else:
                raise FileNotFoundError(
                    f"\n❌ 预训练模型不存在！\n"
                    f"   需要: {pretrain_path}\n"
                    f"   备选: {alt_path} (也不存在)\n"
                    f"   请先运行: python scripts/train_basic_v2.py"
                )

        pretrain = torch.load(pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(pretrain['model_state_dict'])
        print(f"\n📥 已加载 {source_subset} 预训练权重 "
              f"(Epoch {pretrain.get('epoch', '?')})")

    print(f"🔧 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ============================================================
# 8. 训练单个目标域（可独立调用）
# ============================================================
def train_single_target(source_subset, target_subset, device, resume=False,
                       preset='static', adapt_mode='lmmd_semi'):
    """训练 FD001 → target 的迁移模型"""

    # 每个目标域、每种适配模式使用独立的 checkpoint
    prefix = 'static' if preset == 'static' else f'dynatopo_{preset}'
    if preset == 'static':
        checkpoint_path = (
            f'saved_models/original_paper_static/transfer/{adapt_mode}/'
            f'transfer_static_{adapt_mode}_checkpoint_{target_subset}.pt'
        )
    else:
        checkpoint_path = f'saved_models/transfer_{prefix}_{adapt_mode}_checkpoint_{target_subset}.pt'
    if preset == 'static':
        pretrain_path = (
            'saved_models/original_paper_static/stgnn/'
            f'stgnn_static_best_{source_subset}.pt'
        )
    else:
        pretrain_path = f'saved_models/dynatopo_{preset}_best_{source_subset}.pt'

    # ---- 加载数据 ----
    (src_train_loader, src_val_loader,
     tgt_train_loader, tgt_val_loader,
     src_edge, tgt_edge) = load_transfer_data(source_subset, target_subset)

    # ---- 构建模型 + 加载预训练权重 ----
    model = build_and_load_model(source_subset, device, resume, pretrain_path, preset)

    # ---- 损失函数和优化器 ----
    task_loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, nasa_weight=NASA_SCORE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)

    print(f"  任务损失: CombinedLoss (MSE×{MSE_WEIGHT} + NASA×{NASA_SCORE_WEIGHT})")
    print(f"  LMMD λ: {LMMD_LAMBDA}, 目标域任务权重 w: {TGT_TASK_WEIGHT}")
    print(f"  优化器: Adam (lr={LEARNING_RATE * 0.1})")

    # ---- 训练 ----
    model, train_losses, tgt_val_losses = train_transfer(
        model, src_train_loader, src_val_loader,
        tgt_train_loader, tgt_val_loader,
        task_loss_fn, src_edge, tgt_edge, optimizer, device,
        target_subset=target_subset,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
        lmmd_lambda=LMMD_LAMBDA,
        resume=resume,
        checkpoint_path=checkpoint_path,
        prefix=prefix,
        adapt_mode=adapt_mode,
    )

    # ---- 保存日志 ----
    save_training_log(train_losses, tgt_val_losses,
                      source=source_subset, target=target_subset,
                      prefix=prefix, adapt_mode=adapt_mode)

    return model


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='跨工况迁移学习训练 (static / dynatopo)')
    parser.add_argument('--source', type=str, default='FD001', help='源域数据集')
    parser.add_argument('--target', type=str, default='all',
                        help='目标域数据集 (FD002/FD003/FD004 或 all)')
    parser.add_argument('--preset', type=str, default='static',
                        help='模型预设: static / A1B1 / A1B2 / A2B1 / A2B2')
    parser.add_argument('--adapt_mode', type=str, default='lmmd_semi',
                        choices=['lmmd_semi', 'lmmd_uda', 'global_mmd', 'none'],
                        help='域自适应模式')
    parser.add_argument('--resume', action='store_true', help='从 checkpoint 续训')
    args = parser.parse_args()

    source_subset = args.source
    preset = args.preset
    adapt_mode = args.adapt_mode

    # 确定目标域列表
    if args.target == 'all':
        target_list = ['FD002', 'FD003', 'FD004']
    else:
        target_list = [args.target]

    print("=" * 60)
    print(f"  🧪 跨工况迁移学习训练 [{preset}, {adapt_mode}]")
    print(f"  {source_subset} (源域) → {target_list} (目标域)")
    print("=" * 60)

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  训练设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- 逐目标域训练 ----
    for target_subset in target_list:
        print(f"\n{'#'*60}")
        print(f"#  开始训练: {source_subset} → {target_subset} [{preset}, {adapt_mode}]")
        print(f"{'#'*60}")

        train_single_target(source_subset, target_subset, device,
                            resume=args.resume, preset=preset, adapt_mode=adapt_mode)

        print(f"\n🎉 {source_subset} → {target_subset} [{preset}, {adapt_mode}] 训练完成！")

    print(f"\n{'='*60}")
    print(f"  🎉 全部迁移学习训练完成！")
    print(f"  模型保存在: saved_models/transfer_*_best_*.pt")
    print(f"{'='*60}")
