# ============================================================
# configs/dynatopo_config.py —— DynaTopo 实验组合配置
# ============================================================
# 定义动态图生成策略（A）和图融合策略（B）的枚举与组合预设。
# 实验脚本通过 --preset 参数一键切换，无需手动改代码。
#
# 使用方式：
#   from configs.dynatopo_config import EXPERIMENT_MATRIX, get_preset
#   cfg = get_preset("A1B1")  # 相似度生成 + 特征层融合
# ============================================================

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict


# ============================================================
# 类型定义
# ============================================================
GeneratorType = Literal["similarity", "attention"]
FusionType = Literal["feature", "topology"]
AdaptMode = Literal["none", "global_mmd", "lmmd_uda", "lmmd_semi"]


# ============================================================
# 实验配置数据类
# ============================================================

@dataclass
class DynaTopoConfig:
    """单个实验的完整配置"""
    # ---- 实验标识 ----
    name: str                                    # 实验名称（用于日志和输出）
    preset: str                                  # 预设名（如 "A1B1"）

    # ---- 图生成策略 (A) ----
    generator: Literal["similarity", "attention", "none"] = "similarity"
    generator_top_k: int = 20                    # 动态图保留的边数
    generator_hidden: int = 64                   # 生成器内部隐藏维度

    # ---- 图融合策略 (B) ----
    fusion: Literal["feature", "topology", "none"] = "feature"
    fusion_out_dim: int = 64                     # 融合后输出维度

    # ---- 静态图开关 ----
    use_static_graph: bool = True                # 是否保留 Spearman 静态图

    # ---- 工况调制开关（消融用）----
    use_op_modulation: bool = True               # 动态图生成是否受工况参数调制

    # ---- MSTCN/GAT 参数（沿用 config.py 中的值）----
    mstcn_channels: list = field(default_factory=lambda: [32, 64, 128])
    mstcn_kernels: list = field(default_factory=lambda: [3, 5, 7])
    mstcn_dropout: float = 0.2
    gat_hidden: int = 64
    gat_heads: int = 4
    gat_dropout: float = 0.2


@dataclass
class TransferConfig:
    """单个迁移实验的配置"""
    preset: str = ""                             # 预设名
    model_type: Literal["static", "dynatopo"] = "static"  # 模型类型
    generator: Optional[str] = None              # 生成器类型（dynatopo 时使用）
    fusion: Optional[str] = None                 # 融合类型（dynatopo 时使用）
    source: str = "FD001"                        # 源域
    target: str = "FD002"                        # 目标域
    adapt_mode: AdaptMode = "lmmd_semi"          # 域自适应方法
    semi_weight: float = 1.0                     # 目标域任务损失权重


# ============================================================
# 实验矩阵：A × B 全组合
# ============================================================

EXPERIMENT_MATRIX: Dict[str, DynaTopoConfig] = {
    # ===== A1 × B1: 相似度生成 + 特征层融合 =====
    "A1B1": DynaTopoConfig(
        name="A1B1: 相似度生成 × 特征层融合",
        preset="A1B1",
        generator="similarity",
        fusion="feature",
        use_static_graph=True,
    ),

    # ===== A1 × B2: 相似度生成 + 拓扑层融合 =====
    "A1B2": DynaTopoConfig(
        name="A1B2: 相似度生成 × 拓扑层融合",
        preset="A1B2",
        generator="similarity",
        fusion="topology",
        use_static_graph=True,
    ),

    # ===== A2 × B1: 注意力生成 + 特征层融合 =====
    "A2B1": DynaTopoConfig(
        name="A2B1: 注意力生成 × 特征层融合",
        preset="A2B1",
        generator="attention",
        fusion="feature",
        use_static_graph=True,
    ),

    # ===== A2 × B2: 注意力生成 + 拓扑层融合 =====
    "A2B2": DynaTopoConfig(
        name="A2B2: 注意力生成 × 拓扑层融合",
        preset="A2B2",
        generator="attention",
        fusion="topology",
        use_static_graph=True,
    ),
}


# ============================================================
# 消融对照组
# ============================================================

ABLATION_CONFIGS: Dict[str, DynaTopoConfig] = {
    # 仅静态图（= 原 STGNN_Static，作为基线）
    "static_only": DynaTopoConfig(
        name="消融: 仅静态图（原 STGNN 基线）",
        preset="static_only",
        generator="none",
        fusion="none",
        use_static_graph=True,
    ),

    # 仅动态图（无 Spearman 先验）
    "dynamic_only": DynaTopoConfig(
        name="消融: 仅动态图（无 Spearman）",
        preset="dynamic_only",
        generator="similarity",
        fusion="feature",
        use_static_graph=False,
    ),

    # ===== A2B2 组件消融（UDA 场景，FD002）=====
    "A2B2_wo_dynamic": DynaTopoConfig(
        name="消融: A2B2 去动态图（=仅静态图）",
        preset="A2B2_wo_dynamic",
        generator="none",
        fusion="none",
        use_static_graph=True,
    ),
    "A2B2_wo_static": DynaTopoConfig(
        name="消融: A2B2 去静态图（=仅注意力动态图）",
        preset="A2B2_wo_static",
        generator="attention",
        fusion="feature",
        use_static_graph=False,
    ),
    "A2B2_wo_op": DynaTopoConfig(
        name="消融: A2B2 去工况调制（注意力但不看 op）",
        preset="A2B2_wo_op",
        generator="attention",
        fusion="topology",
        use_static_graph=True,
        use_op_modulation=False,
    ),
}


# ============================================================
# 迁移实验预设
# ============================================================

TRANSFER_PRESETS: Dict[str, TransferConfig] = {
    # ===== 静态图：复现论文表5-7 =====
    "static_FD001_to_FD002_none": TransferConfig(
        preset="static_FD002_none",
        model_type="static", target="FD002", adapt_mode="none",
    ),
    "static_FD001_to_FD002_lmmd_uda": TransferConfig(
        preset="static_FD002_lmmd_uda",
        model_type="static", target="FD002", adapt_mode="lmmd_uda",
    ),
    "static_FD001_to_FD002_lmmd_semi": TransferConfig(
        preset="static_FD002_lmmd_semi",
        model_type="static", target="FD002", adapt_mode="lmmd_semi",
    ),
    "static_FD001_to_FD003_lmmd_semi": TransferConfig(
        preset="static_FD003_lmmd_semi",
        model_type="static", target="FD003", adapt_mode="lmmd_semi",
    ),
    "static_FD001_to_FD004_lmmd_semi": TransferConfig(
        preset="static_FD004_lmmd_semi",
        model_type="static", target="FD004", adapt_mode="lmmd_semi",
    ),
    "static_FD001_to_FD002_global_mmd": TransferConfig(
        preset="static_FD002_global_mmd",
        model_type="static", target="FD002", adapt_mode="global_mmd",
    ),

    # ===== 双图模型迁移（后续按需添加）=====
    # "dynatopo_A1B1_FD001_to_FD002_lmmd_semi": TransferConfig(
    #     preset="dynatopo_A1B1_FD002",
    #     model_type="dynatopo", generator="similarity", fusion="feature",
    #     target="FD002", adapt_mode="lmmd_semi",
    # ),
}


# ============================================================
# 工具函数
# ============================================================

def get_experiment_config(preset: str) -> DynaTopoConfig:
    """根据预设名获取实验配置"""
    # 先查主矩阵
    if preset in EXPERIMENT_MATRIX:
        return EXPERIMENT_MATRIX[preset]
    # 再查消融配置
    if preset in ABLATION_CONFIGS:
        return ABLATION_CONFIGS[preset]
    raise ValueError(
        f"未知的预设: '{preset}'。"
        f"可用预设: {list(EXPERIMENT_MATRIX.keys()) + list(ABLATION_CONFIGS.keys())}"
    )


def get_transfer_config(preset: str) -> TransferConfig:
    """根据预设名获取迁移实验配置"""
    if preset in TRANSFER_PRESETS:
        return TRANSFER_PRESETS[preset]
    raise ValueError(
        f"未知的迁移预设: '{preset}'。"
        f"可用预设: {list(TRANSFER_PRESETS.keys())}"
    )


def list_all_presets():
    """列出所有可用预设"""
    print("\n📋 可用实验预设:")
    print("─" * 60)
    print("【4种 A×B 组合】")
    for k, v in EXPERIMENT_MATRIX.items():
        print(f"  {k:<10} — {v.name}")
    print("\n【消融对照】")
    for k, v in ABLATION_CONFIGS.items():
        print(f"  {k:<12} — {v.name}")
    print("\n【迁移实验】")
    for k, v in TRANSFER_PRESETS.items():
        print(f"  {k:<35} — {v.model_type} {v.target} {v.adapt_mode}")


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    list_all_presets()
    print("\n✅ 测试 A1B1 配置:")
    cfg = get_experiment_config("A1B1")
    print(f"  generator={cfg.generator}, fusion={cfg.fusion}, "
          f"use_static={cfg.use_static_graph}")
