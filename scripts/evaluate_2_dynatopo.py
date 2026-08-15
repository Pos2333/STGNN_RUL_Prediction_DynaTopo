# ============================================================
# scripts/evaluate_2_dynatopo.py —— 跨工况评估（static + dynatopo）
# ============================================================
# 对每个模型（static / A1B1 / A1B2 等），在每个目标域（FD002~FD004）上评估：
#   1. 无迁移:     FD001 预训练模型直接测试目标域
#   2. 半监督LMMD: FD001→目标域迁移模型（源域+目标域监督 + LMMD）
#
# 用法:
#   python scripts/evaluate_2_dynatopo.py --preset A1B1       # 单模型
#   python scripts/evaluate_2_dynatopo.py --preset all        # 全部模型
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    BATCH_SIZE, RANDOM_SEED,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from configs.dynatopo_config import get_experiment_config
from core_models.stgnn_static import STGNN_Static
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from utils.metrics import compute_rmse, compute_nasa_score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性推理（保证同模型可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_target_test(target, processed_dir='data/processed'):
    test_path = os.path.join(processed_dir, f'{target}_test.npz')
    graph_path = os.path.join(processed_dir, f'{target}_train_graph.pt')

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试数据不存在: {test_path}")

    test_data = np.load(test_path)
    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = test_data['y']
    ruls = test_data.get('rul_true', y_test)

    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    edge_index = graph['edge_index']

    print(f"  📂 {target} 测试数据: {len(X_test)} 样本")
    return X_test, y_test, ruls, edge_index


def build_model(preset, device):
    """根据预设构建模型（static 或 dynatopo）"""
    if preset == 'static':
        return STGNN_Static(
            num_sensors=14, num_op_settings=3,
            mstcn_channels=MSTCN_NUM_CHANNELS, mstcn_kernels=MSTCN_KERNEL_SIZES,
            mstcn_dropout=MSTCN_DROPOUT,
            gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_HEADS, gat_dropout=GAT_DROPOUT,
            trans_d_model=TRANSFORMER_D_MODEL, trans_nhead=TRANSFORMER_NHEAD,
            trans_num_layers=TRANSFORMER_NUM_LAYERS, trans_dropout=TRANSFORMER_DROPOUT,
            use_transformer=False,
            fc_hidden=FC_HIDDEN_DIM
        ).to(device)
    else:
        cfg = get_experiment_config(preset)
        return STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3,
                              fc_hidden=FC_HIDDEN_DIM).to(device)


@torch.no_grad()
def evaluate(model, X_test, edge_index, device):
    model.eval()
    loader = DataLoader(
        TensorDataset(X_test, torch.zeros(len(X_test))),
        batch_size=BATCH_SIZE, shuffle=False
    )
    preds = []
    for X_b, _ in loader:
        X_b = X_b.to(device)
        pred = model(X_b, edge_index.to(device))
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='A1B1',
                        help='模型预设: static / A1B1 / A1B2 / A2B1 / A2B2 / all')
    args = parser.parse_args()

    if args.preset == 'all':
        presets = ['static', 'A1B1', 'A2B1', 'A2B2']
    else:
        presets = [args.preset]

    print("=" * 70)
    print(f"  📊 跨工况评估（无迁移 vs LMMD半监督 vs 无自适应微调）—— {presets}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}\n")

    targets = ['FD002', 'FD003', 'FD004']
    all_results = {}

    for preset in presets:
        prefix = 'static' if preset == 'static' else f'dynatopo_{preset}'
        # FD001 预训练模型路径
        if preset == 'static':
            pretrain_path = 'saved_models/stgnn_static_best_FD001.pt'
        else:
            pretrain_path = f'saved_models/dynatopo_{preset}_best_FD001.pt'

        print(f"\n{'#'*70}")
        print(f"#  模型: {preset}")
        print(f"{'#'*70}")

        for target in targets:
            X_test, y_test, ruls, edge_index = load_target_test(target)

            # ---- 无迁移：FD001 预训练模型直接测目标域 ----
            no_transfer_rmse = no_transfer_score = None
            if os.path.exists(pretrain_path):
                model = build_model(preset, device)
                ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                preds = evaluate(model, X_test, edge_index, device)
                no_transfer_rmse = float(compute_rmse(preds, y_test))
                no_transfer_score = float(compute_nasa_score(preds, ruls))
                print(f"\n  [{preset}] 无迁移 {target}: "
                      f"RMSE={no_transfer_rmse:.2f}, NASA={no_transfer_score:.1f}")
            else:
                print(f"\n  [{preset}] 无迁移 {target}: ⚠️ 缺少预训练模型 {pretrain_path}")

            # ---- 半监督 LMMD：迁移模型 ----
            transfer_path = f'saved_models/transfer_{prefix}_lmmd_semi_best_{target}.pt'
            semi_rmse = semi_score = None
            if os.path.exists(transfer_path):
                model = build_model(preset, device)
                ckpt = torch.load(transfer_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                preds = evaluate(model, X_test, edge_index, device)
                semi_rmse = float(compute_rmse(preds, y_test))
                semi_score = float(compute_nasa_score(preds, ruls))
                print(f"  [{preset}] 半监督LMMD {target}: "
                      f"RMSE={semi_rmse:.2f}, NASA={semi_score:.1f}")
            else:
                print(f"  [{preset}] 半监督LMMD {target}: ⚠️ 缺少迁移模型 {transfer_path}")

            # ---- 无自适应（none）：目标域有监督微调，作为域自适应基线下限 ----
            none_path = f'saved_models/transfer_{prefix}_none_best_{target}.pt'
            none_rmse = none_score = None
            if os.path.exists(none_path):
                model = build_model(preset, device)
                ckpt = torch.load(none_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                preds = evaluate(model, X_test, edge_index, device)
                none_rmse = float(compute_rmse(preds, y_test))
                none_score = float(compute_nasa_score(preds, ruls))
                print(f"  [{preset}] 无自适应none {target}: "
                      f"RMSE={none_rmse:.2f}, NASA={none_score:.1f}")
            else:
                print(f"  [{preset}] 无自适应none {target}: ⚠️ 缺少迁移模型 {none_path}")

            # ---- 无监督域自适应（lmmd_uda）：目标域无标签，单向 LMMD ----
            uda_path = f'saved_models/transfer_{prefix}_lmmd_uda_best_{target}.pt'
            uda_rmse = uda_score = None
            if os.path.exists(uda_path):
                model = build_model(preset, device)
                ckpt = torch.load(uda_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                preds = evaluate(model, X_test, edge_index, device)
                uda_rmse = float(compute_rmse(preds, y_test))
                uda_score = float(compute_nasa_score(preds, ruls))
                print(f"  [{preset}] 无监督UDA {target}: "
                      f"RMSE={uda_rmse:.2f}, NASA={uda_score:.1f}")
            else:
                print(f"  [{preset}] 无监督UDA {target}: ⚠️ 缺少迁移模型 {uda_path}")

            all_results[f'{preset}_{target}'] = {
                'preset': preset, 'target': target,
                'no_transfer_rmse': no_transfer_rmse,
                'no_transfer_score': no_transfer_score,
                'semi_rmse': semi_rmse,
                'semi_score': semi_score,
                'none_rmse': none_rmse,
                'none_score': none_score,
                'uda_rmse': uda_rmse,
                'uda_score': uda_score,
            }

    # ---- 汇总表格（每个模型/目标域打印 RMSE 与 NASA 两行）----
    print(f"\n{'='*80}")
    print(f"  📋 跨工况评估汇总")
    print(f"{'='*80}")
    print(f"  {'模型':<8} {'目标':<7} {'指标':<6} {'无迁移':>12} {'LMMD-semi':>12} {'none(FT)':>12} {'UDA':>12}")
    print(f"  {'-'*80}")
    for k, r in all_results.items():
        def fmt(v, nd=2):
            return f"{v:.{nd}f}" if v is not None else "—"
        # RMSE 行
        print(f"  {r['preset']:<8} {r['target']:<7} {'RMSE':<6} "
              f"{fmt(r['no_transfer_rmse']):>12} {fmt(r['semi_rmse']):>12} {fmt(r['none_rmse']):>12} "
              f"{fmt(r['uda_rmse']):>12}")
        # NASA Score 行
        print(f"  {'':<8} {'':<7} {'NASA':<6} "
              f"{fmt(r['no_transfer_score'], 1):>12} {fmt(r['semi_score'], 1):>12} "
              f"{fmt(r['none_score'], 1):>12} {fmt(r['uda_score'], 1):>12}")

    # 保存
    os.makedirs('logs/dynatopo', exist_ok=True)
    with open('logs/dynatopo/eval_cross_condition.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n📝 结果已保存至 logs/dynatopo/eval_cross_condition.json")


if __name__ == '__main__':
    main()
