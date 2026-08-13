# ============================================================
# scripts/evaluate_2_dynatopo.py —— 双图模型跨工况评估
# ============================================================
# 评估 dynatopo 迁移模型在 FD002~FD004 上的表现。
#
# 用法:
#   python scripts/evaluate_2_dynatopo.py --preset A1B1
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import BATCH_SIZE, RANDOM_SEED
from configs.dynatopo_config import get_experiment_config
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

    graph = torch.load(graph_path, map_location='cpu')
    edge_index = graph['edge_index']

    print(f"  📂 {target} 测试数据: {len(X_test)} 样本")
    return X_test, y_test, ruls, edge_index


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
    parser.add_argument('--preset', type=str, default='A1B1')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  📊 STGNN_DynaTopo 跨工况评估 [{args.preset}]")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    cfg = get_experiment_config(args.preset)
    targets = ['FD002', 'FD003', 'FD004']
    results = {}

    for target in targets:
        transfer_path = f'saved_models/dynatopo_{args.preset}_transfer_{target}.pt'
        if not os.path.exists(transfer_path):
            print(f"\n  ⚠️  跳过 {target}: 迁移模型不存在 ({transfer_path})")
            continue

        X_test, y_test, ruls, edge_index = load_target_test(target)

        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3).to(device)
        ckpt = torch.load(transfer_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        preds = evaluate(model, X_test, edge_index, device)
        rmse = compute_rmse(preds, y_test)
        score = compute_nasa_score(preds, ruls)

        print(f"\n  {target}: RMSE={rmse:.2f}, NASA Score={score:.1f}")
        results[target] = {'rmse': float(rmse), 'score': float(score)}

    print(f"\n{'='*60}")
    print(f"  📋 跨工况评估汇总 [{args.preset}]")
    print(f"{'='*60}")
    for t, r in results.items():
        print(f"  {t}: RMSE={r['rmse']:.2f}, NASA Score={r['score']:.1f}")


if __name__ == '__main__':
    main()
