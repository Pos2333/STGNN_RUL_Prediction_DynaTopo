# ============================================================
# core_models/stgnn_static.py —— 静态图 STGNN 模型拼装（原论文最终架构）
# ============================================================
# TODO 3: 将 MSTCN + GAT + Transformer 三位一体组装
# TODO 6: 新增消融实验开关 use_mstcn / use_gat / use_transformer
#
# ⚠️ Shape 冲突预警（严格遵循 TODO.md 要求）:
#   数据输入格式: [B, W, 17]  (batch, 窗口, 特征)
#   - MSTCN 需要: [B, 14, W]  (batch, 传感器, 窗口) → permute(0,2,1)
#   - GAT 需要:    [B*14, D]  (所有传感器铺平) + edge_index
#   - Transformer:  [B, W, 14] (batch, 窗口, 传感器) → 时间维 attention
#
# 数据流:
#   ┌──── 输入: [B, W, 17] ────┐
#   │                          │
#   ├→ 操作参数 [B, W, 3]      ├→ 传感器数据 [B, W, 14]
#   │   permute(0,2,1)         │
#   │   → [B, 3, W]            │   ┌→ MSTCN 分支
#   │   → pool → Linear        │   │  permute(0,2,1) → [B, 14, W]
#   │   → [B, 16]              │   │  MSTCN → [B, 14, 128]
#   │                          │   │
#   │                          │   ├→ GAT 分支
#   │                          │   │  reshape [B*14, 128]
#   │                          │   │  GATConv + edge_index → [B*14, 64]
#   │                          │   │  reshape [B, 14, 64]
#   │                          │   │  mean_pool → [B, 64]
#   │                          │   │
#   │                          │   └→ Transformer 分支
#   │                          │      [B, W, 14]
#   │                          │      Transformer → [B, 128]
#   │                          │
#   └──────────────────────────┘
#              ↓
#       Concat: [B, 16+64+128] = [B, 208]
#              ↓
#       FC → [B, 1]  (RUL 预测值)
#
# 消融实验说明（TODO 6）:
#   - use_mstcn=False:     用单层 Conv1d(k=1)+Pool 替代多尺度时序卷积
#   - use_gat=False:       用传感器维度 mean_pool+Linear 替代图注意力
#   - use_transformer=False:用时间维度 mean_pool+Linear 替代全局self-attention
#   每种替代均保持输出维度不变，融合层无需调整
#
# 注意：本文件仅在 TODO 3 使用 MSE+NASA Score 训练。
#       LMMD 损失在 TODO 5 才会加入，此处绝不提前引入！
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_models.mstcn import MSTCN
from core_models.gat import SensorGAT
from core_models.transformer import SensorTransformer


def repeat_edge_index_for_batch(edge_index, batch_size, num_nodes=14):
    """
    为 batch 中的每个样本复制图结构并加上偏移量

    由于我们将 [B, 14, D] reshape 为 [B*14, D]，
    每个样本内的传感器节点索引需要加上样本偏移。

    例如：batch_size=2, num_nodes=14
      样本 0 的节点: 0~13
      样本 1 的节点: 14~27
      edge_index 中的边索引也需要相应偏移

    参数:
        edge_index: [2, E]  单个样本的图边索引
        batch_size: B
        num_nodes:  每个样本的节点数（= NUM_SENSORS = 14）

    返回:
        batched_edge_index: [2, B*E]  所有样本的边索引
    """
    edge_list = []
    for i in range(batch_size):
        offset = i * num_nodes
        edge_list.append(edge_index + offset)
    return torch.cat(edge_list, dim=1)


class STGNN_Static(nn.Module):
    """
    时空图神经网络（静态图版本）—— 原论文最终架构（支持消融实验）

    架构:
      - MSTCN:  提取多尺度局部时序特征
      - GAT:    基于 Spearman 固定拓扑图建模传感器空间依赖关系
      - Transformer: 捕获全局长程时序依赖（v2 论文版默认关闭）

    消融开关（TODO 6）:
      - use_mstcn:      是否使用多尺度时间卷积
      - use_gat:        是否使用图注意力网络
      - use_transformer: 是否使用 Transformer 全局注意力
      关闭任一开关时，该分支被最简操作替代，输出维度不变

    参数:
        num_sensors:       传感器数量（14）
        num_op_settings:   操作参数数量（3）
        mstcn_channels:    MSTCN 各层通道数
        mstcn_kernels:     MSTCN 各层卷积核大小
        mstcn_dropout:     MSTCN dropout
        gat_hidden:        GAT 隐藏层维度
        gat_heads:         GAT 注意力头数
        gat_dropout:       GAT dropout
        trans_d_model:     Transformer 模型维度
        trans_nhead:       Transformer 头数
        trans_num_layers:  Transformer 层数
        trans_dropout:     Transformer dropout
        use_mstcn:         消融开关：是否启用 MSTCN
        use_gat:           消融开关：是否启用 GAT
        use_transformer:   消融开关：是否启用 Transformer
        fc_hidden:         全连接层隐藏维度
    """

    def __init__(self, num_sensors=14, num_op_settings=3,
                 mstcn_channels=None, mstcn_kernels=None, mstcn_dropout=0.2,
                 gat_hidden=64, gat_heads=4, gat_dropout=0.2,
                 trans_d_model=128, trans_nhead=4, trans_num_layers=2, trans_dropout=0.2,
                 use_mstcn=True, use_gat=True, use_transformer=True,
                 fc_hidden=64):
        super(STGNN_Static, self).__init__()

        if mstcn_channels is None:
            mstcn_channels = [32, 64, 128]
        if mstcn_kernels is None:
            mstcn_kernels = [3, 5, 7]

        self.num_sensors = num_sensors
        self.num_op_settings = num_op_settings

        # ---- 消融实验开关 ----
        self.use_mstcn = use_mstcn
        self.use_gat = use_gat
        self.use_transformer = use_transformer

        # ============================================================
        # 操作参数编码分支（始终存在，不参与消融）
        # ============================================================
        self.op_conv = nn.Sequential(
            nn.Conv1d(num_op_settings, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.op_pool = nn.AdaptiveAvgPool1d(1)

        # ============================================================
        # MSTCN 分支 —— 多尺度时序特征
        # 若 use_mstcn=False，用单层 Conv1d(k=1)+Pool 替代
        # ============================================================
        mstcn_out_dim = mstcn_channels[-1]  # 128

        if self.use_mstcn:
            self.mstcn = MSTCN(
                num_sensors=num_sensors,
                num_channels=mstcn_channels,
                kernel_sizes=mstcn_kernels,
                dropout=mstcn_dropout
            )
        else:
            # 最简替代：逐点卷积 + 全局池化，无多尺度感受野
            self.mstcn_simple = nn.Sequential(
                nn.Conv1d(1, mstcn_out_dim, kernel_size=1),
                nn.BatchNorm1d(mstcn_out_dim),
                nn.ReLU()
            )

        # ============================================================
        # GAT 分支 —— 传感器空间依赖
        # 若 use_gat=False，用传感器维度平均池化 + Linear 替代
        # ============================================================
        gat_out_dim = gat_hidden  # 64

        if self.use_gat:
            self.gat = SensorGAT(
                in_channels=mstcn_out_dim,
                hidden_dim=gat_hidden,
                heads=gat_heads,
                dropout=gat_dropout
            )
        else:
            # 最简替代：对传感器维度直接求均值，然后线性投影
            self.gat_simple = nn.Sequential(
                nn.Linear(mstcn_out_dim, mstcn_out_dim // 2),
                nn.ReLU(),
                nn.Linear(mstcn_out_dim // 2, gat_out_dim)
            )

        # ============================================================
        # Transformer 分支 —— 全局时序依赖
        # 若 use_transformer=False，用时间维度平均池化 + Linear 替代
        # ============================================================
        trans_out_dim = trans_d_model  # 128

        if self.use_transformer:
            self.transformer = SensorTransformer(
                input_dim=num_sensors,
                d_model=trans_d_model,
                nhead=trans_nhead,
                num_layers=trans_num_layers,
                dropout=trans_dropout
            )
        else:
            # 最简替代：对时间维度直接求均值，然后线性投影
            self.trans_simple = nn.Sequential(
                nn.Linear(num_sensors, num_sensors * 2),
                nn.ReLU(),
                nn.Linear(num_sensors * 2, trans_out_dim)
            )

        # ============================================================
        # 融合全连接层（始终 16 + 64 + 128 = 208 维输入）
        # ============================================================
        fusion_dim = 16 + gat_out_dim + trans_out_dim  # 208
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, 1),
            nn.ReLU()  # RUL >= 0
        )

    def forward(self, x, edge_index, return_feat=False):
        """
        前向传播（支持消融实验开关）

        参数:
            x:            [B, W, 17]  输入特征（3个操作参数 + 14个传感器）
            edge_index:   [2, E]      图边索引（use_gat=False 时可传 None）
            return_feat:  是否返回融合特征（用于 LMMD 迁移学习）

        返回:
            若 return_feat=False: output [B, 1]  RUL 预测值
            若 return_feat=True:  (output [B, 1], fused_feat [B, 208])
        """
        B, W, _ = x.shape

        # ============================================================
        # 拆分操作参数与传感器数据
        # ============================================================
        op_feat = x[:, :, :self.num_op_settings]        # [B, W, 3]
        sensor_feat = x[:, :, self.num_op_settings:]     # [B, W, 14]

        # ============================================================
        # 分支 1: 操作参数编码（始终执行，不参与消融）
        # ============================================================
        op_out = op_feat.permute(0, 2, 1)   # [B, 3, W]
        op_out = self.op_conv(op_out)        # [B, 16, W]
        op_out = self.op_pool(op_out)        # [B, 16, 1]
        op_out = op_out.squeeze(-1)          # [B, 16]

        # ============================================================
        # 分支 2: MSTCN → GAT（传感器空间-时序联合特征）
        # ============================================================
        # ---- Step 2a: 时序特征提取（MSTCN 或简单替代） ----
        mstcn_in = sensor_feat.permute(0, 2, 1)   # [B, 14, W]

        if self.use_mstcn:
            # 完整 MSTCN：三层多尺度卷积提取局部时序特征
            mstcn_out = self.mstcn(mstcn_in)       # [B, 14, 128]
        else:
            # 消融版：单层逐点卷积 + 全局池化，无多尺度感受野
            B_s, N_s, W_s = mstcn_in.shape
            simple_in = mstcn_in.reshape(B_s * N_s, 1, W_s)   # [B*14, 1, W]
            simple_out = self.mstcn_simple(simple_in)          # [B*14, 128, W]
            simple_out = F.adaptive_avg_pool1d(simple_out, 1)  # [B*14, 128, 1]
            mstcn_out = simple_out.squeeze(-1).reshape(B_s, N_s, -1)  # [B, 14, 128]

        # ---- Step 2b: 传感器空间建模（GAT 或简单替代） ----
        if self.use_gat:
            # 完整 GAT：图注意力聚合邻居传感器信息
            B_sensor, N_sensor, D_sensor = mstcn_out.shape
            gat_in = mstcn_out.reshape(B_sensor * N_sensor, D_sensor)  # [B*14, 128]
            batched_edge = repeat_edge_index_for_batch(edge_index, B, self.num_sensors)
            gat_out = self.gat(gat_in, batched_edge)                    # [B*14, 64]
            gat_out = gat_out.reshape(B, self.num_sensors, -1)          # [B, 14, 64]
            gat_out = gat_out.mean(dim=1)                                # [B, 64]
        else:
            # 消融版：传感器维度直接平均池化 + 线性投影
            gat_out = mstcn_out.mean(dim=1)           # [B, 128]
            gat_out = self.gat_simple(gat_out)         # [B, 64]

        # ============================================================
        # 分支 3: Transformer（全局时序依赖）
        # ============================================================
        if self.use_transformer:
            # 完整 Transformer：全局 self-attention 捕获长程时序依赖
            trans_out = self.transformer(sensor_feat)   # [B, 128]
        else:
            # 消融版：时间维度直接平均池化 + 线性投影
            trans_out = sensor_feat.mean(dim=1)          # [B, 14]
            trans_out = self.trans_simple(trans_out)      # [B, 128]

        # ============================================================
        # 特征融合
        # ============================================================
        fused = torch.cat([op_out, gat_out, trans_out], dim=1)  # [B, 208]

        # ============================================================
        # 全连接输出
        # ============================================================
        output = self.fc(fused)  # [B, 1]

        if return_feat:
            return output, fused
        return output


# ============================================================
# 测试入口 —— 包含消融实验各变体的自测
# ============================================================
if __name__ == '__main__':
    print("🧪 STGNN 完整模型自测（含消融变体）")

    B, W, F = 4, 30, 17
    dummy_x = torch.randn(B, W, F)

    # 模拟图结构（链式边）
    dummy_edge = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ], dtype=torch.long)

    print(f"输入形状: {dummy_x.shape}")
    print(f"边索引形状: {dummy_edge.shape}")

    # ---- 测试四种消融变体 ----
    ablation_configs = {
        "完整 STGNN":        dict(use_mstcn=True,  use_gat=True,  use_transformer=True),
        "无 MSTCN":          dict(use_mstcn=False, use_gat=True,  use_transformer=True),
        "无 GAT":            dict(use_mstcn=True,  use_gat=False, use_transformer=True),
        "无 Transformer":    dict(use_mstcn=True,  use_gat=True,  use_transformer=False),
    }

    for name, cfg in ablation_configs.items():
        print(f"\n{'─'*50}")
        print(f"🔬 {name}: use_mstcn={cfg['use_mstcn']}, "
              f"use_gat={cfg['use_gat']}, use_transformer={cfg['use_transformer']}")

        model = STGNN_Static(
            num_sensors=14, num_op_settings=3,
            mstcn_channels=[32, 64, 128], mstcn_kernels=[3, 5, 7], mstcn_dropout=0.2,
            gat_hidden=64, gat_heads=4, gat_dropout=0.2,
            trans_d_model=128, trans_nhead=4, trans_num_layers=2, trans_dropout=0.2,
            use_mstcn=cfg['use_mstcn'], use_gat=cfg['use_gat'],
            use_transformer=cfg['use_transformer'],
            fc_hidden=64
        )
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 前向传播测试
        edge_input = dummy_edge if cfg['use_gat'] else None
        output = model(dummy_x, edge_input)
        print(f"  输出形状: {output.shape}  ✅ 通过")

        # 测试 return_feat 模式
        output2, feat = model(dummy_x, edge_input, return_feat=True)
        print(f"  融合特征形状: {feat.shape}  ✅ return_feat 通过")

    print(f"\n{'='*50}")
    print("✅ 全部消融变体自测通过！四种模型均可正常前向传播。")
