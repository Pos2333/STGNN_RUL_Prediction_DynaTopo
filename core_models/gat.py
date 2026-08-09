# ============================================================
# core_models/gat.py —— 图注意力网络模块
# ============================================================
# TODO 3: 建模 14 个传感器之间的空间依赖关系
#
# 核心思路：
#   1. 接收 MSTCN 输出的每个传感器的特征向量 [B, N, D_in]
#   2. Reshape 为 [B*N, D_in]，配合 edge_index 送入 GATConv
#   3. 多头注意力自动学习哪些传感器之间应该"互相参考"
#      （例如：温度传感器和压力传感器可能有强关联）
#   4. 输出 reshape 回 [B, N, D_out]，再 mean pool → [B, D_out]
#
# Shape 流转（严格按照 TODO 预警）:
#   输入:  [B, N, D_in]       例: [256, 14, 128]
#   → reshape [B*N, D_in]     例: [3584, 128]
#   → GATConv + edge_index    例: [3584, 256]
#   → reshape [B, N, D_out]   例: [256, 14, 256]
#   → mean(dim=1)             例: [256, 256]
# ============================================================

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SensorGAT(nn.Module):
    """
    传感器图注意力网络

    将 14 个传感器视为图中的 14 个节点，边由 Spearman 相关系数决定。
    通过多头注意力机制，自动学习节点之间的信息传递权重。

    参数:
        in_channels:   输入特征维度（= MSTCN 输出维度）
        hidden_dim:    GAT 隐藏层维度（来自 config.GAT_HIDDEN_DIM）
        heads:         注意力头数（来自 config.GAT_HEADS）
        dropout:       Dropout 比率（来自 config.GAT_DROPOUT）
    """

    def __init__(self, in_channels, hidden_dim=64, heads=4, dropout=0.2):
        super(SensorGAT, self).__init__()

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.out_dim = hidden_dim * heads  # 多头拼接后的输出维度

        # ---- 第一层 GAT：从输入维度映射到 hidden_dim ----
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True  # 多头结果拼接（而非平均），输出维度 = hidden_dim * heads
        )

        # ---- 第二层 GAT：进一步聚合邻居信息 ----
        # 输入是 heads * hidden_dim，输出降回 hidden_dim（用单头做最终融合）
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,       # 单头做最终输出
            dropout=dropout,
            concat=False   # 单头时不拼接，输出维度 = hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        前向传播

        参数:
            x:          [B*N, D_in]  所有样本的所有传感器特征铺平
            edge_index: [2, E]       图结构的边索引（PyG 格式）

        返回:
            out: [B, D_out]  每个样本的图级别特征（mean pool 后）
        """
        # ---- GAT 第一层 ----
        x = self.gat1(x, edge_index)      # [B*N, heads * hidden_dim]
        x = torch.relu(x)
        x = self.dropout(x)

        # ---- GAT 第二层 ----
        x = self.gat2(x, edge_index)      # [B*N, hidden_dim]
        x = torch.relu(x)

        return x  # 返回节点级特征，由 stgnn_full.py 负责 reshape 和 pool


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 GAT 模块自测")

    # 模拟输入
    B, N, D_in = 4, 14, 128
    dummy_x = torch.randn(B * N, D_in)
    print(f"输入形状: {dummy_x.shape}")

    # 模拟图结构：14 个节点的全连接图（去掉自环）
    # 生成简单的链式边结构用于测试
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ], dtype=torch.long)
    print(f"边索引形状: {edge_index.shape}")

    model = SensorGAT(in_channels=D_in, hidden_dim=64, heads=4, dropout=0.2)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    output = model(dummy_x, edge_index)
    print(f"输出形状: {output.shape}")  # 期望: [56, 64] 即 [B*N, hidden_dim]
