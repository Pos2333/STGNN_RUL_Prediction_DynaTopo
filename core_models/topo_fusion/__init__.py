# ============================================================
# core_models/topo_fusion/__init__.py
# 图融合策略子包
# ============================================================
from .base_fusion import BaseTopoFusion
from .feature_fusion import FeatureFusion
from .topology_fusion import TopologyFusion


def get_fusion(fusion_type: str, **kwargs):
    """
    工厂函数：根据类型名返回对应的图融合策略实例

    参数:
        fusion_type: "feature" (B1) 或 "topology" (B2)
        **kwargs:    传递给具体融合器的参数

    返回:
        BaseTopoFusion 子类实例
    """
    if fusion_type == "feature":
        return FeatureFusion(**kwargs)
    elif fusion_type == "topology":
        return TopologyFusion(**kwargs)
    else:
        raise ValueError(
            f"未知的融合类型: '{fusion_type}'。"
            f"可选: 'feature', 'topology'"
        )
