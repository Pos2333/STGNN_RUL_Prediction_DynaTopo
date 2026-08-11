# ============================================================
# scripts/evaluate_1_dynatopo.py —— 双图模型单工况评估
# ============================================================
# 在 FD001 测试集上评估各 A×B 组合模型的预测性能。
#
# 用法:
#   python scripts/evaluate_1_dynatopo.py
# ============================================================

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE, RANDOM_SEED
)
from configs.dynatopo_config import get_experiment_config, EXPERIMENT_MATRIX
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from utils.metrics import compute_rmse, compute_nasa_score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_test_data(subset='FD001', processed_dir='data/processed'):
    """加载预处理好的测试数据"""
    test_path = os.path.join(processed_dir, f'{subset}_test.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')

    test_data = np.load(test_path)
    X_test = test_data['X']
    y_test = test_data['y']
    ruls = test_data.get('rul_true', y_test)

    graph = torch.load(graph_path)
    edge_index = graph['edge_index']

    print(f"📂 测试数据: {len(X_test)} 样本")
    return (torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
            ruls, edge_index)


@torch.no_grad()
def evaluate_model(model, X_test, edge_index, device):
    model.eval()
    loader = DataLoader(
        TensorDataset(X_test, torch.zeros(len(X_test))),
        batch_size=BATCH_SIZE, shuffle=False
    )
    all_preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch, edge_index.to(device))
        all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds)


def main():
    print("=" * 60)
    print("  📊 STGNN_DynaTopo 单工况评估 (FD001)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    X_test, y_test, ruls, edge_index = load_test_data('FD001')

    results = {}

    for preset, cfg in EXPERIMENT_MATRIX.items():
        model_path = f'saved_models/dynatopo_{preset}_best_FD001.pt'
        if not os.path.exists(model_path):
            print(f"\n  ⚠️  跳过 {preset}: 模型不存在 ({model_path})")
            continue

        print(f"\n{'─'*50}")
        print(f"  🔍 {cfg.name}")

        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        preds = evaluate_model(model, X_test, edge_index, device)
        rmse = compute_rmse(preds, y_test.numpy())
        score = compute_nasa_score(preds, ruls)
        params = sum(p.numel() for p in model.parameters())

        print(f"    RMSE: {rmse:.2f}, NASA Score: {score:.1f}, 参数: {params:,}")
        results[preset] = {'rmse': float(rmse), 'score': float(score), 'params': params}

    # 打印汇总表格
    print(f"\n{'='*60}")
    print("  📋 单工况评估汇总")
    print(f"{'='*60}")
    print(f"  {'模型':<20} {'RMSE':>8} {'NASA Score':>12} {'参数':>10}")
    print(f"  {'─'*50}")
    for preset, r in results.items():
        print(f"  {preset:<20} {r['rmse']:>8.2f} {r['score']:>12.1f} {r['params']:>10,}")

    # 保存结果
    os.makedirs('logs/dynatopo', exist_ok=True)
    with open('logs/dynatopo/eval_single_FD001.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📝 结果已保存至 logs/dynatopo/eval_single_FD001.json")


if __name__ == '__main__':
    main()
