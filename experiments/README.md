# 实验隔离区 (experiments/)

本目录用于存放**与主线（seed=42）隔离的探索性实验**，不进入论文最终结论。

## 目录结构

```
experiments/
├── a2_stability/               # A2 稳定性实验脚本
│   ├── config_seeds.py         # 实验配置（seed 列表、模型预设）
│   ├── train_multi_seed.py     # 单次训练脚本（--preset + --seed）
│   ├── run_all.py              # 一键跑完所有 preset × seed 组合
│   └── analyze.py              # 汇总 mean ± std + 箱线图
├── models/                     # 实验模型权重（与主线 saved_models 隔离）
└── logs/                       # 实验日志（与主线 logs 隔离）
```

## 使用方式

```bash
# 方式一：一键跑完所有组合（4 预设 × 3 seed = 12 次训练）
python experiments/a2_stability/run_all.py

# 方式二：手动单次训练
python experiments/a2_stability/train_multi_seed.py --preset A2B1 --seed 114514

# 方式三：分析结果
python experiments/a2_stability/analyze.py
```

## 实验目的

1. 修正 `attention_generator.py` 的 softmax 归一化后，A2 是否公平
2. 多 seed 评估 A1/A2 的指标离散程度（mean ± std）
3. 判断 A1 与 A2 的性能差异是否显著、稳定

## 清理

实验完成后若需恢复干净状态，直接删除整个 `experiments/` 目录即可，
主线代码与主线结果（seed=42）不受任何影响。
