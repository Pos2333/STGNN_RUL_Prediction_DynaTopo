# ============================================================
# core_models/topo_generator/__init__.py
# 动态图生成器子包
# ============================================================
from .base_generator import BaseDynamicGraphGenerator
from .similarity_generator import SimilarityGenerator
from .attention_generator import AttentionGenerator


def get_generator(generator_type: str, **kwargs):
    """
    工厂函数：根据类型名返回对应的动态图生成器实例

    参数:
        generator_type: "similarity" (A1) 或 "attention" (A2)
        **kwargs:       传递给具体生成器的参数

    返回:
        BaseDynamicGraphGenerator 子类实例
    """
    if generator_type == "similarity":
        return SimilarityGenerator(**kwargs)
    elif generator_type == "attention":
        return AttentionGenerator(**kwargs)
    else:
        raise ValueError(
            f"未知的生成器类型: '{generator_type}'。"
            f"可选: 'similarity', 'attention'"
        )
