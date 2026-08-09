[English Version](./README.md)

# 航空发动机剩余寿命（RUL）预测

本仓库为课题 **《基于图神经网络的航空发动机剩余寿命预测方法研究》** 的完整代码实现，基于 NASA C-MAPSS 数据集。同时该仓库在 GitHub 上公开，供学习与参考使用。

> **论文 STGNN 最终架构 = MSTCN + GAT（v2），不使用 Transformer。**
> 详见下方 [模型架构](#模型架构) 与 [版本说明](#版本说明)。

## 项目结构

```
RUL_Prediction/
├── data/                # 数据仓库（raw 原始数据 / processed 预处理数据）
├── core_models/         # 模型零件库（MSTCN, GAT, Transformer, STGNN 拼装）
├── utils/               # 工具箱（数据处理、损失函数、评估指标）
├── configs/             # 全局配置
├── scripts/             # 训练与评估脚本
├── notebooks/           # 可视化分析与图表绘制
├── extracted_pdf/       # 课题论文 PDF 提取内容
├── saved_models/        # 训练好的模型权重
└── logs/                # 运行日志
```

## 环境配置

```bash
# 创建虚拟环境
conda create -n rul_env python=3.10 -y
conda activate rul_env

# 安装依赖
pip install -r requirements.txt
```

## 数据集

NASA C-MAPSS 数据集（FD001 ~ FD004），存放在 `data/raw/` 下。

## 模型架构

课题最终采用的 **STGNN（时空图神经网络）** 由以下模块组成：

- **MSTCN**（多尺度时间卷积网络）：三层堆叠 Conv1d（kernel 3→5→7, channels 32→64→128），参数共享提取 14 个传感器的多尺度时序退化特征
- **GAT**（图注意力网络）：两层多头注意力（4-head → 1-head），基于 Spearman 秩相关系数构建的传感器拓扑图建模空间耦合关系
- **全局时序上下文**：时间维均值池化 + 两层 Linear，补充全局退化趋势信息
- **操作工况编码**：Conv1d 编码 3 个操作参数
- **特征融合**：拼接工况编码、GAT 空间特征与全局时序上下文 → FC → RUL
- **损失函数**：MSE + NASA 非对称评分联合损失
- **迁移学习**：基于 RUL 退化阶段子域划分的 LMMD（局部最大均值差异）跨工况自适应

> 注：代码库中保留了 `Transformer` 模块（`core_models/transformer.py`）及其消融实验开关 `use_transformer`，用于架构探索与对比实验。经消融实验验证，Transformer 分支对最终性能贡献有限，因此课题论文的最终 STGNN 模型**不使用 Transformer**。

## 版本说明

| 版本 | 架构 | 标识 | 说明 |
|------|------|------|------|
| **v2（主线）** | MSTCN + GAT | `_v2` 后缀 | **论文采用的最终架构**，关闭 Transformer 分支 |
| v1（探索） | MSTCN + GAT + Transformer | 无后缀 | 三位一体完整架构，保留作对比参考 |

- `scripts/train_basic_v2.py`、`scripts/evaluate_2_v2.py` — 单工况 v2 训练与评估
- `scripts/ablation_study_v2.py` — v2 消融实验（MSTCN+GAT / 仅MSTCN / 仅GAT / 全关）
- `scripts/train_transfer_v2_*.py` — v2 跨工况迁移实验（无监督UDA / 半监督 / 全局MMD对比）
- 不含 `_v2` 的同名脚本为 v1 版本，包含 Transformer 分支
