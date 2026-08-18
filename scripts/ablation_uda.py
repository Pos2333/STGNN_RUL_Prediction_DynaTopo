# ============================================================
# scripts/ablation_uda.py —— A2B2 组件消融实验（UDA 场景，FD001→FD002）
# ============================================================
# 消融变体：
#   A2B2（完整）         : 注意力动态图 + 拓扑融合 + 静态图 + 工况调制
#   A2B2_wo_dynamic      : 仅静态图（=static_only，消融动态分支）
#   A2B2_wo_static       : 仅注意力动态图（消融静态先验）
#   A2B2_wo_op           : 注意力动态图 + 拓扑融合 + 静态图，但工况调制关闭
#
# 均使用 lmmd_uda 无监督域自适应，FD001 → FD002。
# 每个变体独立训练 + 测试集评估。
# ============================================================

import os
import sys
import json
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    BATCH_SIZE, RANDOM_SEED, MSE_WEIGHT, NASA_SCORE_WEIGHT,
    LMMD_LAMBDA, TGT_TASK_WEIGHT,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from configs.dynatopo_config import get_experiment_config
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from core_models.stgnn_static import STGNN_Static
from utils.loss_functions import CombinedLoss, lmmd_loss
from utils.metrics import compute_rmse, compute_nasa_score
from utils.data_processor import split_by_unit

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 消融变体定义
ABLATION_PRESETS = {
    'A2B2': 'A2B2',
    'wo_dynamic': 'A2B2_wo_dynamic',
    'wo_static': 'A2B2_wo_static',
    'wo_op': 'A2B2_wo_op',
}
# 对应 checkpoint 前缀（用于区分日志/模型）
PREFIX_MAP = {'A2B2': 'abl_A2B2', 'wo_dynamic': 'abl_wodynamic',
              'wo_static': 'abl_wostatic', 'wo_op': 'abl_woop'}


def load_transfer_data(source='FD001', target='FD002', val_ratio=0.15):
    """加载源域和目标域数据，按 unit 分组拆分（与 train_transfer 一致）"""
    src = np.load(os.path.join('data/processed', f'{source}_train.npz'))
    tgt = np.load(os.path.join('data/processed', f'{target}_train.npz'))
    X_src, y_src, u_src = src['X'], src['y'], src['unit']
    X_tgt, y_tgt, u_tgt = tgt['X'], tgt['y'], tgt['unit']

    src_edge = torch.load(os.path.join('data/processed', f'{source}_train_graph.pt'),
                          weights_only=False)['edge_index']
    tgt_edge_path = os.path.join('data/processed', f'{target}_train_graph.pt')
    tgt_edge = (torch.load(tgt_edge_path, weights_only=False)['edge_index']
                if os.path.exists(tgt_edge_path) else src_edge)

    X_src_tr, X_src_va, y_src_tr, y_src_va = split_by_unit(
        X_src, y_src, u_src, val_ratio=val_ratio, random_state=RANDOM_SEED)
    X_tgt_tr, X_tgt_va, y_tgt_tr, y_tgt_va = split_by_unit(
        X_tgt, y_tgt, u_tgt, val_ratio=val_ratio, random_state=RANDOM_SEED)

    def mk_loader(X, y, shuffle=True):
        return DataLoader(TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)),
            batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

    return (mk_loader(X_src_tr, y_src_tr, True),
            mk_loader(X_src_va, y_src_va, False),
            mk_loader(X_tgt_tr, y_tgt_tr, True),
            mk_loader(X_tgt_va, y_tgt_va, False),
            src_edge, tgt_edge)


def build_model(preset, device):
    """构建模型（支持静态/消融预设）"""
    if preset == 'A2B2_wo_dynamic':
        return STGNN_Static(
            num_sensors=14, num_op_settings=3,
            mstcn_channels=MSTCN_NUM_CHANNELS, mstcn_kernels=MSTCN_KERNEL_SIZES,
            mstcn_dropout=MSTCN_DROPOUT,
            gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_HEADS, gat_dropout=GAT_DROPOUT,
            trans_d_model=TRANSFORMER_D_MODEL, trans_nhead=TRANSFORMER_NHEAD,
            trans_num_layers=TRANSFORMER_NUM_LAYERS, trans_dropout=TRANSFORMER_DROPOUT,
            use_transformer=False, fc_hidden=FC_HIDDEN_DIM,
        ).to(device)
    else:
        cfg = get_experiment_config(preset)
        return STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3,
                              fc_hidden=FC_HIDDEN_DIM).to(device)


def train_uda(model, src_tr, tgt_tr, loss_fn, src_edge, tgt_edge, optimizer,
              device, num_epochs=200, patience=20):
    """UDA 训练循环（与 train_transfer.py lmmd_uda 一致）"""
    best_tgt_loss = float('inf')
    best_state = None
    patience_cnt = 0

    src_iter = iter(src_tr)
    tgt_iter = iter(tgt_tr)
    num_batches = max(len(src_tr), len(tgt_tr))

    for epoch in range(num_epochs):
        model.train()
        total_src, total_lmmd = 0.0, 0.0
        for _ in range(num_batches):
            try:
                X_s, y_s = next(src_iter)
            except StopIteration:
                src_iter = iter(src_tr)
                X_s, y_s = next(src_iter)
            try:
                X_t, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_tr)
                X_t, _ = next(tgt_iter)
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t = X_t.to(device)
            pred_s, feat_s = model(X_s, src_edge.to(device), return_feat=True)
            pred_t, feat_t = model(X_t, tgt_edge.to(device), return_feat=True)
            loss_src = loss_fn(pred_s, y_s)
            loss_lmmd = lmmd_loss(feat_s, feat_t, y_s)
            total_loss = loss_src + LMMD_LAMBDA * loss_lmmd
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_src += loss_src.item()
            total_lmmd += loss_lmmd.item()

        # 目标域验证
        model.eval()
        tgt_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in tgt_tr:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b, tgt_edge.to(device))
                tgt_loss += loss_fn(pred, y_b).item() * X_b.size(0)
        tgt_loss /= len(tgt_tr.dataset)

        if tgt_loss < best_tgt_loss:
            best_tgt_loss = tgt_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_test(model, target, edge_index, device):
    """在目标域测试集上评估"""
    data = np.load(os.path.join('data/processed', f'{target}_test.npz'))
    X_test = torch.tensor(data['X'], dtype=torch.float32)
    y_test = data['y']
    ruls = data.get('rul_true', y_test)

    model.eval()
    loader = DataLoader(TensorDataset(X_test, torch.zeros(len(X_test))),
                        batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    for X_b, _ in loader:
        preds.append(model(X_b.to(device), edge_index.to(device)).cpu().numpy())
    preds = np.concatenate(preds)
    return float(compute_rmse(preds, y_test)), float(compute_nasa_score(preds, ruls))


def main():
    print("=" * 60)
    print("  🔬 A2B2 组件消融实验 —— UDA 场景 (FD001 → FD002)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}\n")

    src_tr, src_va, tgt_tr, tgt_va, src_edge, tgt_edge = load_transfer_data()
    loss_fn = CombinedLoss(MSE_WEIGHT, NASA_SCORE_WEIGHT)
    pretrain = torch.load('saved_models/dynatopo_A2B2_best_FD001.pt',
                          map_location=device, weights_only=False)['model_state_dict']

    results = {}
    for label, preset in ABLATION_PRESETS.items():
        print(f"\n{'─'*50}")
        print(f"  🔍 {label} ({preset})")

        model = build_model(preset, device)
        if preset != 'A2B2_wo_dynamic':
            # 动态图模型加载 A2B2 预训练权重（部分参数）
            model.load_state_dict(pretrain, strict=False)
            print(f"  📥 已加载 A2B2 预训练权重（strict=False）")
        else:
            # 静态模型加载静态预训练
            sp = torch.load('saved_models/original_paper_static/stgnn/stgnn_static_best_FD001.pt',
                            map_location=device, weights_only=False)['model_state_dict']
            model.load_state_dict(sp)
            print(f"  📥 已加载 static 预训练权重")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * 0.1)
        model = train_uda(model, src_tr, tgt_tr, loss_fn,
                          src_edge, tgt_edge, optimizer, device)

        test_rmse, test_score = evaluate_test(model, 'FD002', tgt_edge, device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ test RMSE: {test_rmse:.2f}, test NASA: {test_score:.1f}, 参数: {params:,}")

        results[label] = {
            'preset': preset, 'label': label,
            'test_rmse': test_rmse, 'test_nasa': test_score, 'params': params,
        }

    # 打印汇总
    print(f"\n{'='*60}")
    print("  📋 A2B2 组件消融汇总（UDA, FD001→FD002）")
    print(f"{'='*60}")
    print(f"  {'变体':<20} {'test RMSE':>10} {'test NASA':>12} {'参数':>10}")
    print(f"  {'─'*52}")
    for label, r in results.items():
        print(f"  {label:<20} {r['test_rmse']:>10.2f} {r['test_nasa']:>12.1f} {r['params']:>10,}")

    os.makedirs('logs/dynatopo', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'logs/dynatopo/ablation_uda_A2B2_{ts}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📝 结果已保存 → {out}")


if __name__ == '__main__':
    main()