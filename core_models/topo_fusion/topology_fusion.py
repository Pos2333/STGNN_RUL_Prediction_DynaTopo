# ============================================================
# core_models/topo_fusion/topology_fusion.py
# B2: 拓扑层融合 —— 先合并静态边和动态边，再统一 GAT
# ============================================================
# 策略：
#   1. 将 Spearman 静态边和动态生成的边合并为统一的边集
#   2. 去重（同一条边在静态和动态中都存在时只保留一次）
#   3. 统一送入一个 SensorGAT → [B, 64]
#
# 优点：
#   - 静态边和动态边在同一个 GAT 中自然竞争/协作
#   - GAT 注意力可以同时在静态边和动态边上学习权重
#   - 参数更少（只有一个 GAT）
#
# 注意：
#   - 静态边是全局共享的（所有样本相同）
#   - 动态边是每样本独立的（不同工况不同）
#   - 合并时需对静态边做 batch 复制，动态边做 per-sample 提取
# ============================================================

import torch
import torch.nn as nn
from .base_fusion import BaseTopoFusion
from core_models.gat import SensorGAT
from core_models.stgnn_static import repeat_edge_index_for_batch
from core_models.topo_generator.base_generator import adj_matrix_to_edge_index


class TopologyFusion(BaseTopoFusion):
    """
    B2: 拓扑层融合

    将静态边和动态边合并为统一的边集，送入同一个 GAT。

    参数:
        mstcn_out_dim:   MSTCN 输出维度（= 128）
        gat_hidden:       GAT 隐藏维度
        gat_heads:        GAT 头数
        gat_dropout:      GAT dropout
        num_sensors:      传感器数量
        fusion_out_dim:   输出维度
    """

    def __init__(self, mstcn_out_dim=128, gat_hidden=64, gat_heads=4,
                 gat_dropout=0.2, num_sensors=14, fusion_out_dim=64):
        super().__init__(mstcn_out_dim, gat_hidden, gat_heads,
                         gat_dropout, num_sensors, fusion_out_dim)

        # ---- 统一的 GAT（静态边和动态边共享）----
        self.gat_unified = SensorGAT(
            in_channels=mstcn_out_dim,
            hidden_dim=gat_hidden,
            heads=gat_heads,
            dropout=gat_dropout
        )

    def fuse(self, mstcn_out, static_edge_index, adj_dynamic):
        """
        拓扑层融合

        参数:
            mstcn_out:          [B, 14, 128]
            static_edge_index:  [2, E_s]  单样本静态边
            adj_dynamic:        [B, 14, 14]  动态邻接矩阵

        返回:
            fused: [B, fusion_out_dim]
        """
        B = mstcn_out.shape[0]

        # ---- 准备 GAT 输入 ----
        gat_in = mstcn_out.reshape(B * self.num_sensors, -1)

        # ============================================================
        # 合并静态边和动态边
        # ============================================================

        # Step 1: 静态边 batch 复制
        static_batched = repeat_edge_index_for_batch(
            static_edge_index, B, self.num_sensors
        )  # [2, B * E_s]

        # Step 2: 动态边 per-sample 提取
        dynamic_batched = adj_matrix_to_edge_index(
            adj_dynamic, self.num_sensors
        )  # [2, total_dynamic_edges]

        # Step 3: 合并并去重
        if dynamic_batched.numel() > 0:
            all_edges = torch.cat(
                [static_batched, dynamic_batched], dim=1
            ).to(mstcn_out.device)
        else:
            all_edges = static_batched.to(mstcn_out.device)

        # 按列去重（保留唯一的 (src, dst) 对）
        unified_edge = torch.unique(all_edges, dim=1)  # [2, num_unique_edges]

        # ============================================================
        # 统一的 GAT
        # ============================================================
        gat_nodes = self.gat_unified(gat_in, unified_edge)
        # [B*14, gat_hidden] → [B, 14, gat_hidden] → mean pool → [B, gat_hidden]
        gat_out = gat_nodes.reshape(
            B, self.num_sensors, self.gat_hidden
        ).mean(dim=1)  # [B, gat_hidden]

        return gat_out
