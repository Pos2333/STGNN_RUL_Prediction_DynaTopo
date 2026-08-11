# ============================================================
# core_models/topo_fusion/base_fusion.py
# 图融合策略抽象基类
# ============================================================
# 所有融合策略（B1 特征层融合、B2 拓扑层融合）必须继承此类。
#
# 统一接口保证：
#   输入:
#     - mstcn_out:         MSTCN 输出 [B, 14, 128]
#     - static_edge_index:  Spearman 静态图边索引 [2, E_s]
#     - adj_dynamic:        动态图邻接矩阵 [B, 14, 14]
#   输出:
#     - gat_fused:         融合后的空间特征 [B, gat_out_dim]
# ============================================================

import torch
import torch.nn as nn


class BaseTopoFusion(nn.Module):
    """
    图融合策略抽象基类

    子类实现 fuse() 方法，决定静态图和动态图如何结合。

    参数:
        mstcn_out_dim:   MSTCN 输出维度（= 128）
        gat_hidden:       GAT 隐藏层维度（= 64）
        gat_heads:        GAT 注意力头数（= 4）
        gat_dropout:      GAT dropout（= 0.2）
        num_sensors:      传感器数量（= 14）
        fusion_out_dim:   融合后输出维度（= 64）
    """

    def __init__(self, mstcn_out_dim=128, gat_hidden=64, gat_heads=4,
                 gat_dropout=0.2, num_sensors=14, fusion_out_dim=64):
        super().__init__()
        self.mstcn_out_dim = mstcn_out_dim
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.gat_dropout = gat_dropout
        self.num_sensors = num_sensors
        self.fusion_out_dim = fusion_out_dim

    def forward(self, mstcn_out, static_edge_index, adj_dynamic):
        """
        统一前向接口

        参数:
            mstcn_out:          [B, 14, 128]  MSTCN 输出
            static_edge_index:  [2, E_s]      Spearman 静态图边索引（单样本）
            adj_dynamic:        [B, 14, 14]   动态图邻接矩阵（每样本不同）

        返回:
            gat_fused: [B, fusion_out_dim]  融合后的空间特征
        """
        return self.fuse(mstcn_out, static_edge_index, adj_dynamic)

    def fuse(self, mstcn_out, static_edge_index, adj_dynamic):
        """
        融合静态图和动态图特征

        子类必须实现此方法。

        参数:
            mstcn_out:          [B, 14, 128]
            static_edge_index:  [2, E_s]
            adj_dynamic:        [B, 14, 14]

        返回:
            gat_fused: [B, fusion_out_dim]
        """
        raise NotImplementedError("子类必须实现 fuse() 方法")
