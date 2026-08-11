# ============================================================
# notebooks/gat_attention_helper.py
# GAT 注意力权重提取辅助模块
# ============================================================
# 用途：从训练好的 STGNN v2 模型中提取 GAT 层的注意力权重，
#       汇总为 14×14 的传感器间注意力强度矩阵。
#
# 原理：
#   PyG 的 GATConv 在 forward 时支持 return_attention_weights=True，
#   返回 (output, (edge_index, alpha))，其中 alpha 是注意力权重。
#   由于 STGNN 中 batch 维度被 reshape 为 [B*14, D]，每条边在 batch
#   中会被复制 B 次（通过 repeat_edge_index_for_batch）。
#   因此需要按原始节点索引聚合所有 batch 样本中的注意力权重。
#
# 用法：
#   from notebooks.gat_attention_helper import extract_gat_attention
#   attn_matrix = extract_gat_attention(model, X_sample, edge_index, device)
# ============================================================

import torch
import numpy as np


def extract_gat_attention_single(stgnn_model, x_input, edge_index, device='cuda'):
    """
    对单个 batch 样本提取 GAT 第一层的注意力权重

    STGNN 内部数据流:
      1. MSTCN 输出 [B, 14, 128] 
      2. reshape → [B*14, 128] 送入 GAT
      3. GAT 第一层 (4头) 在 edge_index 上计算 attention
      4. 每个样本的图结构由 repeat_edge_index_for_batch 拼合

    因此，edge_index 包含 B 个样本的所有边，注意力权重按边顺序排列。

    参数:
        stgnn_model: 训练好的 STGNN 模型
        x_input:     单 batch 输入 [B, 30, 17]
        edge_index:  原始图边索引 [2, E]（单样本）
        device:      计算设备

    返回:
        attn_weights: [B, E] 每个样本每条边的注意力权重（4头平均）
    """
    stgnn_model.eval()
    B = x_input.shape[0]

    # ---- Step 1: 拆分操作参数和传感器数据（同 STGNN.forward） ----
    op_feat = x_input[:, :, :3]           # [B, W, 3]
    sensor_feat = x_input[:, :, 3:]        # [B, W, 14]

    # ---- Step 2: MSTCN 提取时序特征（同 STGNN.forward） ----
    mstcn_in = sensor_feat.permute(0, 2, 1)    # [B, 14, W]
    mstcn_out = stgnn_model.mstcn(mstcn_in)     # [B, 14, 128]

    # ---- Step 3: 准备 GAT 输入 ----
    B_s, N_s, D_s = mstcn_out.shape
    gat_in = mstcn_out.reshape(B_s * N_s, D_s)  # [B*14, 128]

    # 构造 batch 级 edge_index
    from core_models.stgnn_static import repeat_edge_index_for_batch
    batched_edge = repeat_edge_index_for_batch(edge_index, B, 14).to(device)

    # ---- Step 4: 手动执行 GAT 第一层（带注意力权重提取） ----
    with torch.no_grad():
        # GATConv.forward(x, edge_index, return_attention_weights=True)
        # 返回: (out, (edge_index_out, alpha))
        # ⚠️ GATConv 默认 add_self_loops=True，会自动为每个节点添加自环，
        #    因此返回的 alpha 边数 = 原始边数 + num_nodes
        # alpha shape: [B * (E + num_nodes), heads]
        out1, (ei_out, alpha) = stgnn_model.gat.gat1(
            gat_in, batched_edge, return_attention_weights=True
        )
        # 计算每条样本的实际边数（含自环）
        num_edges_total = alpha.shape[0]
        num_edges_per_sample = num_edges_total // B
        alpha = alpha.reshape(B, num_edges_per_sample, -1)  # [B, E+14, 4]
        alpha_mean = alpha.mean(dim=-1)  # [B, E+14]  4头平均

    # 返回扩展后的单样本边索引（de-offset，含自环）
    # ei_out 包含了所有样本的边（含偏移量 + 自环），需要取第一个样本的
    ei_expanded = ei_out[:, :num_edges_per_sample].clone()
    # 去掉可能的偏移（第一个样本偏移为0）
    ei_expanded = ei_expanded % 14

    return alpha_mean, ei_expanded


def aggregate_to_matrix(attn_weights_batch, edge_index_expanded, num_nodes=14):
    """
    将批量注意力权重聚合为 num_nodes × num_nodes 的矩阵

    由于 STGNN 中每个样本有独立的注意力权重，这里对所有样本
    取中位数（比均值更鲁棒），然后按 (src, dst) 填充矩阵。

    参数:
        attn_weights_batch:  [B, E'] 每个样本每条边的注意力权重（含自环）
        edge_index_expanded: [2, E'] 单样本扩展边索引（含自环）
        num_nodes:           节点数（14）

    返回:
        attn_matrix: [num_nodes, num_nodes] numpy 矩阵
    """
    # 对所有 batch 样本取中位数
    attn_median = attn_weights_batch.median(dim=0).values  # [E']
    attn_median = attn_median.cpu().numpy()
    ei = edge_index_expanded.cpu().numpy()

    attn_matrix = np.zeros((num_nodes, num_nodes))
    for e in range(len(attn_median)):
        src, dst = int(ei[0, e]), int(ei[1, e])
        # 累加（同一边可能有多头/多样本的多次观测）
        attn_matrix[src, dst] = attn_median[e]

    return attn_matrix


def extract_gat_attention_matrix(stgnn_model, data_loader, edge_index, device='cuda',
                                  max_batches=20):
    """
    完整提取流程：从 DataLoader 中取多个 batch，聚合所有注意力权重
    到 14×14 矩阵。

    参数:
        stgnn_model:  训练好的 STGNN 模型
        data_loader:  测试数据的 DataLoader（不打乱）
        edge_index:   原始图边索引 [2, E]
        device:       计算设备
        max_batches:  最多处理多少个 batch（控制计算量）

    返回:
        attn_matrix:  [14, 14] numpy 矩阵，attn[i,j] = 节点i对节点j的注意力强度
        attn_counts:  [14, 14] 每个 (i,j) 对被观测到的次数
    """
    all_weights = []
    edge_index = edge_index.to(device)
    edge_index_expanded_global = None  # 保存第一轮的扩展边索引

    for batch_idx, (X_batch, _) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        X_batch = X_batch.to(device)

        attn_batch, ei_exp = extract_gat_attention_single(
            stgnn_model, X_batch, edge_index, device
        )  # attn_batch: [B, E'], ei_exp: [2, E']
        all_weights.append(attn_batch.cpu())
        if edge_index_expanded_global is None:
            edge_index_expanded_global = ei_exp.cpu()

    all_weights = torch.cat(all_weights, dim=0)  # [total_B, E']

    # 聚合
    attn_matrix = aggregate_to_matrix(all_weights, edge_index_expanded_global)

    return attn_matrix


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 GAT 注意力提取辅助模块自测")
    print("  该模块需配合训练好的 STGNN 模型使用。")
    print("  请从 plot_ch5_gat_attention.py 中调用。")
