# ============================================================
# core_models/stgnn_dynatopo.py —— 双图 STGNN（静态图 + 工况驱动动态图）
# ============================================================
# 在原有 MSTCN+GAT 架构基础上，新增工况感知的动态传感器图生成分支。
# 改动仅限 MSTCN→GAT 之间的两个插入点：
#   - 插入点 A: 动态图生成（similarity / attention）
#   - 插入点 B: 双图融合（feature 各自GAT / topology 合并边）
#
# 架构其余部分（MSTCN、工况编码、全局上下文、FC融合层）完全复用。
#
# 数据流:
#   ┌──── 输入: [B, W, 17] ────┐
#   │                          │
#   ├→ 操作参数 [B, W, 3]      ├→ 传感器数据 [B, W, 14]
#   │   permute(0,2,1)         │
#   │   → [B, 3, W]            │   ┌→ MSTCN 分支
#   │   → Conv → pool          │   │  permute(0,2,1) → [B, 14, W]
#   │   → [B, 16]              │   │  MSTCN → [B, 14, 128]
#   │                          │   │
#   │                          │   ├→ 动态图生成 (A)  🆕
#   │   op_feat ──────────────→│   │  工况调制 → adj_dynamic [B,14,14]
#   │                          │   │
#   │                          │   └→ 双图融合 (B)  🆕
#   │                          │      静态edge_index + 动态adj
#   │                          │         → GAT融合 → [B, 64]
#   │                          │
#   │                          │   └→ 全局上下文
#   │                          │      传感器时间维 mean→ [B,14]
#   │                          │      Linear → [B, 128]
#   │                          │
#   └──────────────────────────┘
#              ↓
#       Concat: [B, 16+64+128] = [B, 208]
#              ↓
#       FC → [B, 1]  (RUL 预测值)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_models.mstcn import MSTCN
from core_models.topo_generator import get_generator
from core_models.topo_fusion import get_fusion


class STGNN_DynaTopo(nn.Module):
    """
    双图时空图神经网络（配置驱动，支持多种 A×B 组合）

    参数:
        config:          DynaTopoConfig 实验配置
        num_sensors:     传感器数量（14）
        num_op_settings: 操作参数数量（3）
        fc_hidden:       全连接层隐藏维度（64）
    """

    def __init__(self, config, num_sensors=14, num_op_settings=3, fc_hidden=64):
        super().__init__()

        self.num_sensors = num_sensors
        self.num_op_settings = num_op_settings
        self.config = config

        # ============================================================
        # 操作参数编码分支（复用原 STGNN_Static 结构）
        # ============================================================
        self.op_conv = nn.Sequential(
            nn.Conv1d(num_op_settings, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.op_pool = nn.AdaptiveAvgPool1d(1)

        # ============================================================
        # MSTCN 分支 —— 多尺度时序特征（复用原结构）
        # ============================================================
        self.mstcn = MSTCN(
            num_sensors=num_sensors,
            num_channels=config.mstcn_channels,
            kernel_sizes=config.mstcn_kernels,
            dropout=config.mstcn_dropout
        )
        mstcn_out_dim = config.mstcn_channels[-1]  # 128

        # ============================================================
        # 动态图生成器 (A) 🆕
        # ============================================================
        self.use_dynamic = config.generator != "none"
        if self.use_dynamic:
            self.topo_generator = get_generator(
                config.generator,
                sensor_dim=mstcn_out_dim,
                op_dim=num_op_settings,
                num_sensors=num_sensors,
                top_k=config.generator_top_k,
                hidden_dim=config.generator_hidden
            )
        else:
            self.topo_generator = None

        # ============================================================
        # 双图融合模块 (B) 🆕
        # ============================================================
        self.use_static = config.use_static_graph

        if self.use_static and self.use_dynamic:
            # 双图模式：使用融合策略
            self.topo_fusion = get_fusion(
                config.fusion,
                mstcn_out_dim=mstcn_out_dim,
                gat_hidden=config.gat_hidden,
                gat_heads=config.gat_heads,
                gat_dropout=config.gat_dropout,
                num_sensors=num_sensors,
                fusion_out_dim=config.fusion_out_dim
            )
        elif self.use_static:
            # 仅静态图模式（= 原 STGNN_Static，用于消融）
            from core_models.gat import SensorGAT
            self.gat_static_only = SensorGAT(
                in_channels=mstcn_out_dim,
                hidden_dim=config.gat_hidden,
                heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            self.topo_fusion = None
            self._gat_out_dim = config.gat_hidden  # 64
        elif self.use_dynamic:
            # 仅动态图模式（消融用）
            from core_models.gat import SensorGAT
            self.gat_dynamic_only = SensorGAT(
                in_channels=mstcn_out_dim,
                hidden_dim=config.gat_hidden,
                heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            self.topo_fusion = None
            self._gat_out_dim = config.gat_hidden
        else:
            raise ValueError("至少需要启用静态图或动态图之一！")

        if self.topo_fusion is not None:
            self._gat_out_dim = config.fusion_out_dim

        # ============================================================
        # 全局时序上下文分支（复用原结构）
        # ============================================================
        trans_out_dim = 128
        self.trans_simple = nn.Sequential(
            nn.Linear(num_sensors, num_sensors * 2),
            nn.ReLU(),
            nn.Linear(num_sensors * 2, trans_out_dim)
        )

        # ============================================================
        # 融合全连接层（始终 16 + 64 + 128 = 208 维输入）
        # ============================================================
        fusion_dim = 16 + self._gat_out_dim + trans_out_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, 1),
            nn.ReLU()  # RUL >= 0
        )

    def forward(self, x, static_edge_index, return_feat=False):
        """
        前向传播

        参数:
            x:                  [B, W, 17]  输入特征
            static_edge_index:  [2, E]      静态图边索引（单样本）
            return_feat:        是否返回融合特征（用于 LMMD）

        返回:
            output [B, 1] 或 (output, fused_feat)
        """
        B, W, _ = x.shape

        # ============================================================
        # 拆分操作参数与传感器数据
        # ============================================================
        op_feat = x[:, :, :self.num_op_settings]      # [B, W, 3]
        sensor_feat = x[:, :, self.num_op_settings:]   # [B, W, 14]

        # ============================================================
        # 分支 1: 操作参数编码
        # ============================================================
        op_out = op_feat.permute(0, 2, 1)   # [B, 3, W]
        op_out = self.op_conv(op_out)        # [B, 16, W]
        op_out = self.op_pool(op_out)        # [B, 16, 1]
        op_out = op_out.squeeze(-1)          # [B, 16]

        # ============================================================
        # 分支 2: MSTCN → 图生成 → 图融合
        # ============================================================
        mstcn_in = sensor_feat.permute(0, 2, 1)   # [B, 14, W]
        mstcn_out = self.mstcn(mstcn_in)           # [B, 14, 128]

        # ---- 动态图生成 ----
        adj_dynamic = None
        if self.use_dynamic:
            adj_dynamic = self.topo_generator(mstcn_out, op_feat)

        # ---- 图融合 / GAT ----
        if self.topo_fusion is not None:
            # 双图融合模式
            gat_out = self.topo_fusion(mstcn_out, static_edge_index, adj_dynamic)
        elif self.use_static:
            # 仅静态图（消融模式）
            from core_models.stgnn_static import repeat_edge_index_for_batch
            gat_in = mstcn_out.reshape(B * self.num_sensors, -1)
            batched_edge = repeat_edge_index_for_batch(
                static_edge_index, B, self.num_sensors
            ).to(x.device)
            gat_nodes = self.gat_static_only(gat_in, batched_edge)
            gat_out = gat_nodes.reshape(B, self.num_sensors, -1).mean(dim=1)
        elif self.use_dynamic:
            # 仅动态图（消融模式）
            from core_models.topo_generator.base_generator import adj_matrix_to_edge_index
            gat_in = mstcn_out.reshape(B * self.num_sensors, -1)
            dynamic_edge = adj_matrix_to_edge_index(adj_dynamic, self.num_sensors).to(x.device)
            gat_nodes = self.gat_dynamic_only(gat_in, dynamic_edge)
            gat_out = gat_nodes.reshape(B, self.num_sensors, -1).mean(dim=1)

        # ============================================================
        # 分支 3: 全局时序上下文
        # ============================================================
        trans_out = sensor_feat.mean(dim=1)      # [B, 14]
        trans_out = self.trans_simple(trans_out)  # [B, 128]

        # ============================================================
        # 特征融合 + 全连接输出
        # ============================================================
        fused = torch.cat([op_out, gat_out, trans_out], dim=1)  # [B, 208]
        output = self.fc(fused)  # [B, 1]

        if return_feat:
            return output, fused
        return output


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from configs.dynatopo_config import get_experiment_config

    print("🧪 STGNN_DynaTopo 自测\n")

    B, W = 4, 30
    dummy_x = torch.randn(B, W, 17)
    dummy_edge = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ], dtype=torch.long)

    print(f"输入形状: {dummy_x.shape}")
    print(f"静态边: {dummy_edge.shape[1]} 条")

    # 测试所有 A×B 组合
    for preset in ["A1B1", "A1B2", "A2B1", "A2B2"]:
        cfg = get_experiment_config(preset)
        print(f"\n{'─'*50}")
        print(f"🔬 {cfg.name}")
        print(f"   generator={cfg.generator}, fusion={cfg.fusion}, "
              f"static={cfg.use_static_graph}")

        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3)
        params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {params:,}")

        output = model(dummy_x, dummy_edge)
        print(f"   输出形状: {output.shape}  ✅")

        # 测试 return_feat 模式
        out2, feat = model(dummy_x, dummy_edge, return_feat=True)
        print(f"   融合特征形状: {feat.shape}  ✅")

    # 测试消融变体
    print(f"\n{'='*50}")
    print("🔬 消融变体测试")
    from configs.dynatopo_config import ABLATION_CONFIGS

    for preset, cfg in ABLATION_CONFIGS.items():
        print(f"\n  {cfg.name}")
        model = STGNN_DynaTopo(cfg, num_sensors=14, num_op_settings=3)
        params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {params:,}")

        output = model(dummy_x, dummy_edge)
        print(f"  输出形状: {output.shape}  ✅")

    print(f"\n{'='*50}")
    print("✅ 全部测试通过！")
