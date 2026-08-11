# ============================================================
# core_models/topo_fusion/feature_fusion.py
# B1: 特征层融合 —— 静态图和动态图各自过 GAT，然后融合特征
# ============================================================
# 策略：
#   1. 静态图 GAT: Spearman edge_index → SensorGAT → [B, 64]
#   2. 动态图 GAT: 动态 adj → SensorGAT → [B, 64]
#   3. 拼接 [B, 128] → 可学习融合层 → [B, 64]
#
# 优点：
#   - 两路独立建模，互不干扰
#   - 消融实验清晰：关闭任一路不影响另一路
#   - 实现简单，参数可控
# ============================================================

import torch
import torch.nn as nn
from .base_fusion import BaseTopoFusion
from core_models.gat import SensorGAT
from core_models.stgnn_static import repeat_edge_index_for_batch
from core_models.topo_generator.base_generator import adj_matrix_to_edge_index


class FeatureFusion(BaseTopoFusion):
    """
    B1: 特征层融合

    两个独立的 GAT 分别在静态图和动态图上运行，
    然后在特征层面做融合。

    参数:
        mstcn_out_dim:   MSTCN 输出维度（= 128）
        gat_hidden:       GAT 隐藏维度
        gat_heads:        GAT 头数
        gat_dropout:      GAT dropout
        num_sensors:      传感器数量
        fusion_out_dim:   融合输出维度
    """

    def __init__(self, mstcn_out_dim=128, gat_hidden=64, gat_heads=4,
                 gat_dropout=0.2, num_sensors=14, fusion_out_dim=64):
        super().__init__(mstcn_out_dim, gat_hidden, gat_heads,
                         gat_dropout, num_sensors, fusion_out_dim)

        # ---- 静态图 GAT（= 原 SensorGAT，不做任何修改）----
        self.gat_static = SensorGAT(
            in_channels=mstcn_out_dim,
            hidden_dim=gat_hidden,
            heads=gat_heads,
            dropout=gat_dropout
        )

        # ---- 动态图 GAT（与静态 GAT 结构相同，参数独立）----
        self.gat_dynamic = SensorGAT(
            in_channels=mstcn_out_dim,
            hidden_dim=gat_hidden,
            heads=gat_heads,
            dropout=gat_dropout
        )

        # ---- 特征融合层 ----
        # 输入: 两个 GAT 输出拼接 [B, gat_hidden + gat_hidden] = [B, 128]
        self.fusion_layer = nn.Sequential(
            nn.Linear(gat_hidden * 2, gat_hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gat_hidden * 2, fusion_out_dim)
        )

    def fuse(self, mstcn_out, static_edge_index, adj_dynamic):
        """
        特征层融合

        参数:
            mstcn_out:          [B, 14, 128]
            static_edge_index:  [2, E_s]  单样本静态边
            adj_dynamic:        [B, 14, 14]  动态邻接矩阵

        返回:
            fused: [B, fusion_out_dim]
        """
        B = mstcn_out.shape[0]

        # ---- 准备 GAT 输入：reshape [B, 14, D] → [B*14, D] ----
        gat_in = mstcn_out.reshape(B * self.num_sensors, -1)

        # ============================================================
        # 分支 1: 静态图 GAT
        # ============================================================
        static_batched_edge = repeat_edge_index_for_batch(
            static_edge_index, B, self.num_sensors
        ).to(mstcn_out.device)

        gat_static_nodes = self.gat_static(gat_in, static_batched_edge)
        # [B*14, gat_hidden] → [B, 14, gat_hidden] → mean pool → [B, gat_hidden]
        gat_static_out = gat_static_nodes.reshape(
            B, self.num_sensors, self.gat_hidden
        ).mean(dim=1)  # [B, gat_hidden]

        # ============================================================
        # 分支 2: 动态图 GAT
        # ============================================================
        dynamic_batched_edge = adj_matrix_to_edge_index(
            adj_dynamic, self.num_sensors
        ).to(mstcn_out.device)

        gat_dynamic_nodes = self.gat_dynamic(gat_in, dynamic_batched_edge)
        gat_dynamic_out = gat_dynamic_nodes.reshape(
            B, self.num_sensors, self.gat_hidden
        ).mean(dim=1)  # [B, gat_hidden]

        # ============================================================
        # 特征融合
        # ============================================================
        fused = self.fusion_layer(
            torch.cat([gat_static_out, gat_dynamic_out], dim=1)
        )  # [B, fusion_out_dim]

        return fused
