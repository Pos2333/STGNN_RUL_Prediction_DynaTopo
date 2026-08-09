# ============================================================
# core_models/mstcn.py —— 多尺度时间卷积模块
# ============================================================
# TODO 3: 从传感器时序数据中提取多尺度局部特征
#
# 核心思路：
#   1. 将 [B, N, W] reshape 为 [B*N, 1, W]，每个传感器独立处理
#   2. 三层堆叠 Conv1d，kernel_size 逐层增大（3→5→7），
#      每层感受野不同，自然形成多尺度特征提取
#   3. 最后 AdaptiveAvgPool1d 将时间维压缩为一个特征向量
#   4. Reshape 回 [B, N, D_out]，每个传感器得到一个 D_out 维特征
#
# Shape 流转（严格按照 TODO 预警）:
#   输入:  [B, N, W]      例: [256, 14, 30]
#   → reshape [B*N, 1, W]  例: [3584, 1, 30]
#   → 3层 Conv1d          例: [3584, 128, 30]
#   → AdaptiveAvgPool1d    例: [3584, 128, 1]
#   → squeeze + reshape    例: [256, 14, 128]
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSTCN(nn.Module):
    """
    多尺度时间卷积网络

    采用堆叠式结构，每层使用不同的卷积核大小，逐层扩大感受野。
    最浅层（k=3）捕捉短期局部模式，中间层（k=5）捕捉中期趋势，
    最深层（k=7）捕捉长期退化规律。三层叠加后，输出特征包含了
    多种时间尺度的信息。

    参数:
        num_sensors:    传感器数量（默认 14，来自 config.NUM_SENSORS）
        num_channels:   各层输出通道数列表（来自 config.MSTCN_NUM_CHANNELS）
        kernel_sizes:   各层卷积核大小列表（来自 config.MSTCN_KERNEL_SIZES）
        dropout:        Dropout 比率（来自 config.MSTCN_DROPOUT）
    """

    def __init__(self, num_sensors=14, num_channels=None,
                 kernel_sizes=None, dropout=0.2):
        super(MSTCN, self).__init__()

        if num_channels is None:
            num_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.num_sensors = num_sensors
        self.out_channels = num_channels[-1]  # 最终输出通道数 = 128

        # ---- 动态构建堆叠卷积层 ----
        layers = []
        in_ch = 1  # 每个传感器作为独立的单变量时间序列

        for i, (out_ch, k) in enumerate(zip(num_channels, kernel_sizes)):
            padding = k // 2  # same padding，保持时间长度不变
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        参数:
            x: [B, N, W]  传感器特征，B=batch, N=传感器数, W=窗口长度

        返回:
            out: [B, N, D]  每个传感器的时序特征向量，D=out_channels
        """
        B, N, W = x.shape

        # ---- Step 1: 将每个传感器拆成独立的单变量时间序列 ----
        # [B, N, W] → [B*N, 1, W]
        x = x.reshape(B * N, 1, W)

        # ---- Step 2: 三层堆叠卷积，提取多尺度时序特征 ----
        # [B*N, 1, W] → [B*N, D, W]
        x = self.conv_stack(x)

        # ---- Step 3: 全局平均池化，将整个时间窗口压缩为一个向量 ----
        # [B*N, D, W] → [B*N, D, 1]
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)  # [B*N, D]

        # ---- Step 4: 恢复传感器维度 ----
        # [B*N, D] → [B, N, D]
        out = x.reshape(B, N, self.out_channels)

        return out


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 MSTCN 模块自测")

    # 模拟输入: [batch=4, 传感器=14, 窗口=30]
    dummy_input = torch.randn(4, 14, 30)
    print(f"输入形状: {dummy_input.shape}")

    model = MSTCN(num_sensors=14, num_channels=[32, 64, 128],
                  kernel_sizes=[3, 5, 7], dropout=0.2)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # 期望: [4, 14, 128]
