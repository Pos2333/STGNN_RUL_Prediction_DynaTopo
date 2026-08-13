# ============================================================
# core_models/topo_generator/attention_generator.py
# A2: 多头注意力 + 工况偏置 动态图生成器
# ============================================================
# 核心思路：
#   1. 将传感器特征通过 Q/K 投影，计算多头注意力分数
#   2. 多头平均得到传感器间关联强度矩阵
#   3. 用工况编码学习偏置项（不同工况改变基线关联强度）
#   4. softmax 归一化（标准注意力实现，消除分数尺度漂移）
#   5. Top-K 稀疏化，保留最强的 K 条边
#
# 为什么选多头注意力：
#   - 比相似度更灵活：不同"头"关注不同语义维度的关联
#   - 与 GAT 中的注意力不同：此处的注意力用于"决定是否连边"，
#     而非"在已有边上分配权重"
#   - 工况偏置提供了结构化的先验调节
#   - softmax 归一化保证分数有界 [0,1]，跨样本尺度一致，
#     使 Top-K 选边稳定可复现
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_generator import BaseDynamicGraphGenerator


class AttentionGenerator(BaseDynamicGraphGenerator):
    """
    基于多头注意力的动态图生成器 (A2)

    流程:
      sensor_feat [B,14,128] → Q/K 投影 → 多头注意力分数 [B,14,14,heads]
                                             ↓ 多头平均 → [B,14,14]
      op_feat [B,W,3] → Conv1d编码 → 工况偏置 ──→ 加到 logits 上
                                             ↓
                                       softmax 归一化 → [B,14,14]
                                             ↓
                                       Top-K 稀疏化 → [B,14,14] 0/1

    参数:
        sensor_dim:   传感器特征维度（= 128）
        op_dim:       操作参数维度（= 3）
        num_sensors:  传感器数量（= 14）
        top_k:        保留的边数
        hidden_dim:   内部投影维度
        num_heads:    注意力头数（= 4）
    """

    def __init__(self, sensor_dim=128, op_dim=3, num_sensors=14,
                 top_k=20, hidden_dim=64, num_heads=4):
        super().__init__(sensor_dim, op_dim, num_sensors, top_k, hidden_dim)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # ---- Q/K 投影（多头）----
        self.q_proj = nn.Linear(sensor_dim, hidden_dim)
        self.k_proj = nn.Linear(sensor_dim, hidden_dim)

        # ---- 工况编码器 ----
        self.op_encoder = nn.Sequential(
            nn.Conv1d(op_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # ---- 工况 → 逐传感器对偏置 ----
        # 输出 N*N 的偏置矩阵，平铺为向量后 reshape
        self.op_to_bias = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_sensors * num_sensors)
        )

    def compute_pairwise_scores(self, sensor_feat, op_feat):
        """
        计算传感器对之间的注意力关联分数

        参数:
            sensor_feat: [B, 14, D]
            op_feat:      [B, W, 3]

        返回:
            scores: [B, 14, 14]
        """
        B, N, D = sensor_feat.shape

        # ---- Step 1: 多头 Q/K 投影 ----
        # Q: [B, N, hidden_dim] → [B, N, heads, head_dim]
        Q = self.q_proj(sensor_feat).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(sensor_feat).view(B, N, self.num_heads, self.head_dim)

        # ---- Step 2: 计算多头注意力分数 ----
        # einsum: b=batch, n/m=传感器, h=头, d=维度
        # [B,N,h,d] × [B,M,h,d] → [B,N,M,h]
        attn = torch.einsum('bnhd,bmhd->bnmh', Q, K)
        # 多头平均
        attn = attn.mean(dim=-1)  # [B, N, N]
        # 缩放（防止内积过大导致 softmax 饱和）
        attn = attn / (self.head_dim ** 0.5)

        # ---- Step 3: 工况偏置 + softmax 归一化 ----
        op_enc = self.op_encoder(op_feat.permute(0, 2, 1))  # [B, 16, 1]
        op_enc = op_enc.squeeze(-1)                          # [B, 16]
        bias = self.op_to_bias(op_enc).view(B, N, N)        # [B, N, N]

        # logits = 注意力分数 + 工况偏置
        logits = attn + bias

        # softmax 归一化（标准 Transformer 注意力实现）：
        #   - 每个源节点 i 对所有目标节点 j 的分数归一化为概率分布
        #   - 分数有界 [0,1]，跨样本尺度一致，Top-K 选边稳定
        #   - 保序性：softmax 单调递增，不改变边的相对排序
        scores = F.softmax(logits, dim=-1)

        return scores
