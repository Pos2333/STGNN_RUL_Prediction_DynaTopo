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
import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WINDOW_SIZE, NUM_FEATURES, BATCH_SIZE, RANDOM_SEED,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from configs.dynatopo_config import get_experiment_config, EXPERIMENT_MATRIX
from core_models.stgnn_static import STGNN_Static
from core_models.stgnn_dynatopo import STGNN_DynaTopo
from utils.metrics import compute_rmse, compute_nasa_score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性推理（保证同模型可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_test_data(subset='FD001', processed_dir='data/processed'):
    """加载预处理好的测试数据"""
    test_path = os.path.join(processed_dir, f'{subset}_test.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')

    test_data = np.load(test_path)
    X_test = test_data['X']
    y_test = test_data['y']
    ruls = test_data.get('true_rul', y_test)

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


def find_latest_log(pattern, dir='logs/dynatopo', fallback_dir='logs'):
    """查找最新的训练日志文件"""
    for d in [dir, fallback_dir]:
        files = sorted(glob.glob(os.path.join(d, pattern)))
        if files:
            return files[-1]
    return None


def read_val_metrics_from_log(log_path):
    """从训练日志中读取验证集指标"""
    with open(log_path) as f:
        data = json.load(f)
    return {
        'val_loss': data.get('best_val_loss'),
        'val_rmse': data.get('best_val_rmse'),
        'val_nasa_score': data.get('best_val_nasa_score'),
        'epochs': data.get('num_epochs'),
    }


def main():
    print("=" * 70)
    print("  📊 STGNN 全面评估 —— FD001 测试集 + 验证集指标")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}\n")

    X_test, y_test, ruls, edge_index = load_test_data('FD001')

    results = {}

    # ================================================================
    # 1. 静态基线
    # ================================================================
    model_path = 'saved_models/original_paper_static/stgnn/stgnn_static_best_FD001.pt'
    log_path = find_latest_log('stgnn_static_FD001_*.json')
    val_info = read_val_metrics_from_log(log_path) if log_path else {}

    if os.path.exists(model_path):
        print(f"{'─'*50}")
        print(f"  🔍 静态基线 (STGNN_Static)")

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
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        preds = evaluate_model(model, X_test, edge_index, device)
        params = sum(p.numel() for p in model.parameters())

        results['static'] = {
            'test_rmse': float(compute_rmse(preds, y_test.numpy())),
            'test_nasa': float(compute_nasa_score(preds, ruls)),
            'params': params,
            **val_info,
        }
        print(f"    test RMSE={results['static']['test_rmse']:.2f}, "
              f"test NASA={results['static']['test_nasa']:.1f}")

    # ================================================================
    # 2. 四组双图模型
    # ================================================================
    for preset, cfg in EXPERIMENT_MATRIX.items():
        model_path = f'saved_models/dynatopo_{preset}_best_FD001.pt'
        log_path = find_latest_log(f'{preset}_FD001_*.json')
        val_info = read_val_metrics_from_log(log_path) if log_path else {}

        if not os.path.exists(model_path):
            print(f"\n  ⚠️  跳过 {preset}: 模型不存在")
            continue

        print(f"\n{'─'*50}")
        print(f"  🔍 {cfg.name}")

        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        preds = evaluate_model(model, X_test, edge_index, device)
        params = sum(p.numel() for p in model.parameters())

        results[preset] = {
            'test_rmse': float(compute_rmse(preds, y_test.numpy())),
            'test_nasa': float(compute_nasa_score(preds, ruls)),
            'params': params,
            **val_info,
        }
        print(f"    test RMSE={results[preset]['test_rmse']:.2f}, "
              f"test NASA={results[preset]['test_nasa']:.1f}")

    # ================================================================
    # 3. 综合对比表
    # ================================================================
    print(f"\n{'='*80}")
    print("  📋 综合对比：val RMSE / val NASA / test RMSE / test NASA / 参数量")
    print(f"{'='*80}")

    hdr = (f"  {'模型':<10} {'val RMSE':>10} {'val NASA':>10} "
           f"{'test RMSE':>10} {'test NASA':>10} {'参数量':>10}")
    print(hdr)
    print(f"  {'─'*64}")

    for name, r in results.items():
        vrmse = r.get('val_rmse', '—')
        vnasa = r.get('val_nasa_score', '—')
        vrmse_str = f"{vrmse:.2f}" if isinstance(vrmse, (int, float)) else str(vrmse)
        vnasa_str = f"{vnasa:.1f}" if isinstance(vnasa, (int, float)) else str(vnasa)
        print(f"  {name:<10} {vrmse_str:>10} {vnasa_str:>10} "
              f"{r['test_rmse']:>10.2f} {r['test_nasa']:>10.1f} {r['params']:>10,}")

    # 保存
    os.makedirs('logs/dynatopo', exist_ok=True)
    with open('logs/dynatopo/eval_full_FD001.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📝 结果已保存至 logs/dynatopo/eval_full_FD001.json")


if __name__ == '__main__':
    main()
