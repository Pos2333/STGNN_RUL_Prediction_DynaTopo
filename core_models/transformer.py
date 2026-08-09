# ============================================================
# core_models/transformer.py —— 全局时序依赖模块
# ============================================================
# TODO 3: 捕获时间维度的全局长程依赖
#
# 核心思路：
#   1. 将传感器特征视为长度为 W 的序列，每个时间步有 N 个传感器值
#   2. 通过 Linear 将 N 维投影到 d_model 维
#   3. 加上可学习的 Positional Encoding（让模型知道时间顺序）
#   4. TransformerEncoder 做全局 self-attention：
#      任意两个时间步之间都可以直接交互，弥补 MSTCN 感受野有限的问题
#   5. Mean pool 将 W 个时间步压缩为一个全局特征向量
#
# Shape 流转:
#   输入:  [B, W, N]         例: [256, 30, 14]
#   → Linear(N, d_model)     例: [256, 30, 128]
#   → + PositionalEncoding   例: [256, 30, 128]
#   → TransformerEncoder     例: [256, 30, 128]
#   → mean(dim=1)            例: [256, 128]
# ============================================================

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    给每个时间步加上一个唯一的"位置签名"，让 Transformer 知道
    "第 5 个时间步"和"第 25 个时间步"在序列中的相对/绝对位置。

    使用固定的 sin/cos 编码（无需学习参数），公式来自
    "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        """
        参数:
            d_model: 模型维度（需与 Transformer 的 d_model 一致）
            max_len: 最大序列长度（预计算足够多的位置）
            dropout: 位置编码后的 dropout
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: 用于不同维度的频率缩放
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # 注册为 buffer（不参与梯度更新，但随模型保存/加载）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: [B, seq_len, d_model]
        返回:
            x + pe: [B, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class SensorTransformer(nn.Module):
    """
    传感器时序 Transformer

    在时间维度上做全局 self-attention，捕获 MSTCN 无法覆盖的
    长程退化趋势（如从早期运行到临近失效的整体模式变化）。

    参数:
        input_dim:   输入特征维度（= NUM_SENSORS = 14）
        d_model:     Transformer 内部维度（来自 config.TRANSFORMER_D_MODEL）
        nhead:       多头注意力头数（来自 config.TRANSFORMER_NHEAD）
        num_layers:  Encoder 层数（来自 config.TRANSFORMER_NUM_LAYERS）
        dropout:     Dropout 比率（来自 config.TRANSFORMER_DROPOUT）
    """

    def __init__(self, input_dim=14, d_model=128, nhead=4,
                 num_layers=2, dropout=0.2):
        super(SensorTransformer, self).__init__()

        self.d_model = d_model

        # ---- 输入投影：将 N 维传感器值映射到 d_model 维 ----
        self.input_proj = nn.Linear(input_dim, d_model)

        # ---- 位置编码 ----
        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,  # 输入格式 [B, seq_len, d_model]
            dim_feedforward=d_model * 4  # FFN 隐藏层维度 = 4 * d_model
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        参数:
            x: [B, W, N]  传感器特征，W=窗口长度(30), N=传感器数(14)

        返回:
            out: [B, d_model]  全局时序特征向量
        """
        # ---- Step 1: 输入投影 ----
        # [B, W, N] → [B, W, d_model]
        x = self.input_proj(x) * math.sqrt(self.d_model)

        # ---- Step 2: 位置编码 ----
        x = self.pos_encoder(x)

        # ---- Step 3: Transformer Encoder ----
        # [B, W, d_model] → [B, W, d_model]
        x = self.transformer_encoder(x)

        # ---- Step 4: 时间维全局平均池化 ----
        # [B, W, d_model] → [B, d_model]
        out = x.mean(dim=1)

        return out


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 Transformer 模块自测")

    # 模拟输入: [batch=4, 窗口=30, 传感器=14]
    dummy_input = torch.randn(4, 30, 14)
    print(f"输入形状: {dummy_input.shape}")

    model = SensorTransformer(
        input_dim=14, d_model=128, nhead=4, num_layers=2, dropout=0.2
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # 期望: [4, 128]
