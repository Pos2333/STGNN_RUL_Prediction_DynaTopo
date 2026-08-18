# ============================================================
# core_models/topo_generator/similarity_generator.py
# A1: 余弦相似度 + 工况调制 动态图生成器
# ============================================================
# 核心思路：
#   1. 将传感器特征投影到低维空间
#   2. 计算传感器对之间的余弦相似度矩阵
#   3. 用工况编码调制相似度（不同工况下"相似"的标准不同）
#   4. Top-K 稀疏化，保留最强的 K 条边
#
# 为什么选余弦相似度：
#   - 直观可解释："两个传感器的退化模式很像 → 它们应该互相通信"
#   - 计算效率高，14 个节点的小图完全够用
#   - 工况调制提供了对不同运行状态的适应性
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_generator import BaseDynamicGraphGenerator


class SimilarityGenerator(BaseDynamicGraphGenerator):
    """
    基于余弦相似度的动态图生成器 (A1)

    流程:
      sensor_feat [B,14,128] → 投影到 hidden_dim → 余弦相似度 [B,14,14]
                                                    ↓
      op_feat [B,W,3] → Conv1d编码 → 工况缩放因子 ──→ 调制相似度
                                                    ↓
                                              Top-K 稀疏化 → [B,14,14] 0/1

    参数:
        sensor_dim:   传感器特征维度（= 128，MSTCN 输出）
        op_dim:       操作参数维度（= 3）
        num_sensors:  传感器数量（= 14）
        top_k:        保留的边数
        hidden_dim:   内部投影维度
    """

    def __init__(self, sensor_dim=128, op_dim=3, num_sensors=14,
                 top_k=20, hidden_dim=64, use_op_modulation=True):
        super().__init__(sensor_dim, op_dim, num_sensors, top_k, hidden_dim)
        self.use_op_modulation = use_op_modulation

        # ---- 传感器特征投影（降维后做相似度计算更稳定）----
        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ---- 工况编码器：将 [B, W, 3] 压缩为 [B, 16] ----
        self.op_encoder = nn.Sequential(
            nn.Conv1d(op_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 16, 1]
        )

        # ---- 工况 → 逐传感器缩放因子 ----
        # 不同工况下，每个传感器维度的"敏感度"不同
        self.op_to_scale = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sensors)
        )

    def compute_pairwise_scores(self, sensor_feat, op_feat):
        """
        计算传感器对之间的关联分数矩阵

        参数:
            sensor_feat: [B, 14, D]  传感器特征
            op_feat:      [B, W, 3]  操作参数

        返回:
            scores: [B, 14, 14]  关联分数（未归一化）
        """
        B, N, D = sensor_feat.shape

        # ---- Step 1: 传感器特征投影 ----
        proj_feat = self.sensor_proj(sensor_feat)  # [B, 14, hidden_dim]

        # ---- Step 2: 余弦相似度矩阵 ----
        # 对每个传感器向量做 L2 归一化
        proj_norm = F.normalize(proj_feat, p=2, dim=-1)  # [B, 14, hidden]
        # 批量矩阵乘法: [B,14,H] × [B,H,14] → [B,14,14]
        sim_matrix = torch.bmm(proj_norm, proj_norm.transpose(1, 2))

        # ---- Step 3: 工况调制 ----
        if self.use_op_modulation:
            # 将工况编码为逐传感器缩放因子
            op_enc = self.op_encoder(op_feat.permute(0, 2, 1))  # [B, 16, 1]
            op_enc = op_enc.squeeze(-1)                          # [B, 16]
            sensor_scale = torch.sigmoid(self.op_to_scale(op_enc))  # [B, 14]

            # 用逐传感器缩放因子调制相似度矩阵
            sim_matrix = sim_matrix * sensor_scale.unsqueeze(2)
            sim_matrix = sim_matrix * sensor_scale.unsqueeze(1)

        return sim_matrix
