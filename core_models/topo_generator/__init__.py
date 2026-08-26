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
        **kwargs:       传递给具体生成器的参数（含 use_op_modulation）

    返回:
        BaseDynamicGraphGenerator 子类实例
    """
    # 默认从 kwargs 中提取 use_op_modulation，不传则默认 True
    use_op = kwargs.pop('use_op_modulation', True)
    if generator_type == "similarity":
        return SimilarityGenerator(use_op_modulation=use_op, **kwargs)
    elif generator_type == "attention":
        return AttentionGenerator(use_op_modulation=use_op, **kwargs)
    else:
        raise ValueError(
            f"未知的生成器类型: '{generator_type}'。"
            f"可选: 'similarity', 'attention'"
        )
