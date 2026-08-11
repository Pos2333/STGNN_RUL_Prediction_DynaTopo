# ============================================================
# core_models/topo_generator/base_generator.py
# 动态图生成器抽象基类
# ============================================================
# 所有动态图生成器（A1 相似度法、A2 注意力法等）必须继承此类，
# 实现 compute_pairwise_scores() 方法。
#
# 统一的前向接口保证：
#   输入: sensor_feat [B, 14, D], op_feat [B, W, 3]
#   输出: adj_dynamic [B, 14, 14]  (0/1 二值邻接矩阵)
# ============================================================

import torch
import torch.nn as nn


class BaseDynamicGraphGenerator(nn.Module):
    """
    动态图生成器抽象基类

    子类只需实现 compute_pairwise_scores()，
    基类负责：去自环 → Top-K 稀疏化 → 输出二值邻接矩阵。

    这种设计确保所有生成器都真正改变了图的边结构（而不是仅调权重），
    从而与 GAT 本身的注意力机制形成本质区别。

    参数:
        sensor_dim:    传感器特征维度（= MSTCN 输出维度，默认 128）
        op_dim:        操作参数维度（= 3）
        num_sensors:   传感器数量（= 14）
        top_k:         保留的最强边数
        hidden_dim:    内部投影维度
    """

    def __init__(self, sensor_dim=128, op_dim=3, num_sensors=14,
                 top_k=20, hidden_dim=64):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.op_dim = op_dim
        self.num_sensors = num_sensors
        self.top_k = top_k
        self.hidden_dim = hidden_dim

    def forward(self, sensor_feat, op_feat):
        """
        统一前向接口

        参数:
            sensor_feat: [B, 14, D]  MSTCN 输出的传感器时序特征
            op_feat:      [B, W, 3]  原始操作参数（工况信息）

        返回:
            adj_dynamic: [B, 14, 14]  动态生成的二值邻接矩阵
                         adj[b, i, j] = 1 表示在样本 b 中传感器 i→j 有边
        """
        # Step 1: 计算传感器对之间的原始关联分数（子类实现）
        score_matrix = self.compute_pairwise_scores(
            sensor_feat, op_feat
        )  # [B, 14, 14]

        # Step 2: 去除自环（传感器自己和自己的边无意义）
        score_matrix = self._remove_self_loops(score_matrix)

        # Step 3: Top-K 稀疏化 —— 确保真正改变了边的结构
        adj = self._topk_sparsify(score_matrix)  # [B, 14, 14]  0/1

        return adj

    def compute_pairwise_scores(self, sensor_feat, op_feat):
        """
        计算传感器对之间的关联分数矩阵

        子类必须实现此方法。

        参数:
            sensor_feat: [B, 14, D]
            op_feat:      [B, W, 3]

        返回:
            scores: [B, 14, 14]  分数越高 → 越应该连边
        """
        raise NotImplementedError(
            "子类必须实现 compute_pairwise_scores() 方法"
        )

    def _remove_self_loops(self, score_matrix):
        """将自环（对角线）的分数设为 -inf，确保不会被选入 Top-K"""
        B = score_matrix.shape[0]
        for b in range(B):
            score_matrix[b].fill_diagonal_(-float('inf'))
        return score_matrix

    def _topk_sparsify(self, score_matrix):
        """
        保留 top_k 条最强的边，其余断开

        这保证了动态图真正改变了邻接结构：
        - 不同样本的 top_k 边不同（因为分数矩阵不同）
        - 不同工况下 top_k 边不同（因为工况影响了分数）
        - 边的集合与 Spearman 静态图不完全重叠
        """
        B, N, _ = score_matrix.shape
        adj = torch.zeros_like(score_matrix)
        flat = score_matrix.view(B, -1)  # [B, N*N]
        _, top_indices = torch.topk(flat, self.top_k, dim=1)  # [B, top_k]
        for b in range(B):
            adj[b].view(-1)[top_indices[b]] = 1.0
        return adj


# ============================================================
# 工具函数：将邻接矩阵转为 PyG 的 edge_index 格式
# ============================================================
def adj_matrix_to_edge_index(adj_matrix, num_nodes=14):
    """
    将 [B, 14, 14] 的邻接矩阵转为 PyG 可用的 batch edge_index

    与 stgnn_static.py 中的 repeat_edge_index_for_batch 类似，
    但这里是从每样本不同的邻接矩阵中提取边。

    参数:
        adj_matrix: [B, num_nodes, num_nodes]  二值邻接矩阵
        num_nodes:  每个样本的节点数

    返回:
        edge_index: [2, total_edges]  所有样本的边索引（带偏移）
    """
    B = adj_matrix.shape[0]
    edge_list = []
    for b in range(B):
        # 找出样本 b 中值为 1 的位置
        edges = torch.nonzero(adj_matrix[b], as_tuple=False)  # [E_b, 2]
        if edges.numel() > 0:
            # 加上样本偏移量
            edges = edges.t() + b * num_nodes  # [2, E_b]
            edge_list.append(edges)
    if len(edge_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=adj_matrix.device)
    return torch.cat(edge_list, dim=1)  # [2, total_edges]
