# 多 seed 稳定性实验 (experiments/a2_stability)

本目录用于多随机种子（multi-seed）稳定性实验：在修复数据泄漏（按发动机分组拆分）、
bias 正初始化等代码问题后，通过多个随机种子重复训练，报告各模型
（static / A1B1 ~ A2B2）验证集与测试集的 mean ± std，为任务 4 的模型选择
提供统计学证据，并可直接用于论文。

> ⚠️ 注意：`experiments/logs/` 与 `experiments/models/` 均被 `.gitignore` 忽略，
> 不会同步到 GitHub。云端（AutoDL）pull 后为干净状态，需重新运行训练并保留输出。

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
# 方式一：一键跑完所有组合（5 预设 × 5 seed = 25 次训练）
#         支持断点续跑：已存在日志的组合自动跳过（云端 SSH 断线后重跑可续上）
python experiments/a2_stability/run_all.py

# 方式二：手动单次训练
python experiments/a2_stability/train_multi_seed.py --preset A2B1 --seed 114514

# 方式三：分析结果
python experiments/a2_stability/analyze.py
```

## 实验目的
多 seed（5 个随机种子）评估 static / A1B1 ~ A2B2 的指标离散程度（mean ± std）
2. 判断各模型在验证集上的差异是否显著、稳定（用于任务 4 的模型选择）
3. 结果以 mean ± std 形式输出，可直接用于论文（mean ± std）
3. 判断 A1 与 A2 的性能差异是否显著、稳定

## 清理

实验完成后若需恢复干净状态，直接删除整个 `experiments/` 目录即可，
主线代码与主线结果（seed=42）不受任何影响。
