# ============================================================
# scripts/evaluate_2_v2.py —— 跨工况迁移实验分析（v2：无 Transformer）
# ============================================================
# TODO 5 动作2：对比三种策略的 STGNN (v2: MSTCN + GAT)
# 在 FD002、FD003、FD004 三个目标数据集上的预测性能
# 消融实验表明无 Transformer 变体性能更优
#
# 对比方案：
#   A. 无迁移:      FD001 训练模型直接测试
#   B. UDA 无监督:  仅源域监督 + LMMD 对齐
#   C. 半监督:      源域+目标域监督 + LMMD 对齐
#
# 用法：
#   python scripts/evaluate_2_v2.py
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
    BATCH_SIZE, RANDOM_SEED,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT,
    FC_HIDDEN_DIM
)
from core_models.stgnn_static import STGNN_Static
from utils.metrics import compute_rmse, compute_nasa_score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# GPU 确定性推理（保证同模型可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 1. 加载测试数据
# ============================================================
def load_test_data(subset, processed_dir='data/processed'):
    """加载指定数据集的测试数据和图结构"""
    test_path = os.path.join(processed_dir, f'{subset}_test.npz')
    graph_path = os.path.join(processed_dir, f'{subset}_train_graph.pt')

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试数据: {test_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"找不到图结构: {graph_path}")

    test_data = np.load(test_path)
    X_test = test_data['X']
    y_test = test_data['y']

    graph = torch.load(graph_path)
    edge_index = graph['edge_index']

    print(f"\n📂 {subset} 测试数据: {len(X_test)} 个样本, "
          f"图边数: {edge_index.shape[1]}")

    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, drop_last=False)

    return loader, edge_index, y_test


# ============================================================
# 2. 创建 STGNN 模型
# ============================================================
def build_model(device):
    """创建与训练时结构一致的 STGNN 模型"""
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
    return model


# ============================================================
# 3. 在单个测试集上评估一个模型
# ============================================================
def evaluate_on_subset(model, test_loader, edge_index, device, model_name):
    """
    在测试集上运行模型并返回指标

    参数:
        model:      已加载权重的 STGNN 模型
        test_loader:测试 DataLoader
        edge_index: 图结构
        device:     设备
        model_name: 模型名称（用于打印）

    返回:
        rmse, score
    """
    print(f"\n  🔍 {model_name}")

    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            edge_d = edge_index.to(device)

            y_pred = model(X_batch, edge_d)
            y_pred_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_batch.numpy())

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    rmse = compute_rmse(y_pred_all, y_true_all)
    score = compute_nasa_score(y_pred_all, y_true_all)

    print(f"    RMSE: {rmse:.2f}, NASA Score: {score:.2f}")

    return rmse, score


# ============================================================
# 4. 主评估流程
# ============================================================
def run_evaluation(target_subsets=None):
    """
    在所有目标数据集上对比三种策略

    参数:
        target_subsets: 目标数据集列表，默认 ['FD002', 'FD003', 'FD004']
    """
    if target_subsets is None:
        target_subsets = ['FD002', 'FD003', 'FD004']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")

    # 收集所有结果
    results = {}

    for target in target_subsets:
        print(f"\n{'='*60}")
        print(f"  评估目标域: {target}")
        print(f"{'='*60}")

        # 加载测试数据
        test_loader, edge_index, y_true = load_test_data(target)

        # ---- A. 无迁移：FD001 预训练模型直接测试 ----
        model_no_transfer = build_model(device)
        pretrain_path = 'saved_models/original_paper_static/stgnn/stgnn_static_best_FD001.pt'
        if os.path.exists(pretrain_path):
            ckpt = torch.load(pretrain_path, map_location=device)
            model_no_transfer.load_state_dict(ckpt['model_state_dict'])
            rmse_a, score_a = evaluate_on_subset(
                model_no_transfer, test_loader, edge_index, device,
                f"无迁移 (FD001→{target})"
            )
        else:
            print(f"  ⚠️  未找到 {pretrain_path}，跳过")
            rmse_a, score_a = None, None

        # ---- B. UDA 无监督：FD001→target 迁移模型（仅 LMMD，不用目标域标签） ----
        uda_path = (
            'saved_models/original_paper_static/transfer/lmmd_uda/'
            f'transfer_static_lmmd_uda_best_{target}.pt'
        )
        model_uda = build_model(device)

        if os.path.exists(uda_path):
            ckpt_u = torch.load(uda_path, map_location=device)
            model_uda.load_state_dict(ckpt_u['model_state_dict'])
            rmse_uda, score_uda = evaluate_on_subset(
                model_uda, test_loader, edge_index, device,
                f"UDA 无监督 (FD001→{target}, 仅LMMD)"
            )
        else:
            print(f"  ⚠️  未找到 UDA 模型 {uda_path}，请先运行 train_transfer.py --adapt_mode lmmd_uda")
            rmse_uda, score_uda = None, None

        # ---- C. 半监督：FD001→target 迁移模型（源域+目标域监督 + LMMD） ----
        transfer_path = (
            'saved_models/original_paper_static/transfer/lmmd_semi/'
            f'transfer_static_lmmd_semi_best_{target}.pt'
        )
        model_transfer = build_model(device)

        if os.path.exists(transfer_path):
            ckpt_t = torch.load(transfer_path, map_location=device)
            model_transfer.load_state_dict(ckpt_t['model_state_dict'])
            rmse_b, score_b = evaluate_on_subset(
                model_transfer, test_loader, edge_index, device,
                f"半监督 (FD001→{target}, LMMD+目标域标签)"
            )
        else:
            print(f"  ⚠️  未找到迁移模型 {transfer_path}，请先运行 train_transfer.py")
            print(f"      提示: 需分别训练 FD001→{target} 的迁移模型")
            rmse_b, score_b = None, None

        results[target] = {
            'no_transfer': {'rmse': rmse_a, 'score': score_a},
            'uda':          {'rmse': rmse_uda, 'score': score_uda},
            'semi_supervised': {'rmse': rmse_b, 'score': score_b},
        }

    # ---- 打印汇总表格 ----
    print(f"\n{'='*80}")
    print(f"  📊 跨工况迁移实验汇总（无迁移 vs UDA 无监督 vs 半监督）")
    print(f"{'='*80}")
    print(f"  {'数据集':<10} {'方法':<22} {'RMSE':>10} {'NASA Score':>15}")
    print(f"  {'-'*57}")

    for target, res in results.items():
        if res['no_transfer']['rmse'] is not None:
            print(f"  {target:<10} {'无迁移':<22} "
                  f"{res['no_transfer']['rmse']:>10.2f} "
                  f"{res['no_transfer']['score']:>15.2f}")
        if res['uda']['rmse'] is not None:
            print(f"  {target:<10} {'UDA 无监督(LMMD)':<22} "
                  f"{res['uda']['rmse']:>10.2f} "
                  f"{res['uda']['score']:>15.2f}")
        if res['semi_supervised']['rmse'] is not None:
            print(f"  {target:<10} {'半监督(LMMD+目标标签)':<22} "
                  f"{res['semi_supervised']['rmse']:>10.2f} "
                  f"{res['semi_supervised']['score']:>15.2f}")

    print(f"{'='*80}")

    # ---- 保存结果到 JSON ----
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/transfer_eval_{timestamp}.json'

    # 将 None 转为字符串以便 JSON 序列化
    json_results = {}
    for target, res in results.items():
        json_results[target] = {
            'no_transfer': {
                'rmse': float(res['no_transfer']['rmse']) if res['no_transfer']['rmse'] is not None else None,
                'score': float(res['no_transfer']['score']) if res['no_transfer']['score'] is not None else None,
            },
            'uda': {
                'rmse': float(res['uda']['rmse']) if res['uda']['rmse'] is not None else None,
                'score': float(res['uda']['score']) if res['uda']['score'] is not None else None,
            },
            'semi_supervised': {
                'rmse': float(res['semi_supervised']['rmse']) if res['semi_supervised']['rmse'] is not None else None,
                'score': float(res['semi_supervised']['score']) if res['semi_supervised']['score'] is not None else None,
            },
        }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    print(f"\n📝 评估结果已保存 → {log_path}")
    print(f"\n🎉 TODO 5 动作2完成！跨工况迁移实验评估完毕。")


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print(f"  🧪 跨工况迁移实验分析 (v2) —— 无迁移 vs UDA vs 半监督")
    print("=" * 60)
    run_evaluation()
