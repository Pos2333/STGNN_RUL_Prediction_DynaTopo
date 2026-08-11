[English Version](./README.md)

# 航空发动机剩余寿命（RUL）预测

本仓库为课题 **《基于图神经网络的航空发动机剩余寿命预测方法研究》** 的完整代码实现，基于 NASA C-MAPSS 数据集。同时该仓库在 GitHub 上公开，供学习与参考使用。

> **论文 STGNN 最终架构 = MSTCN + GAT（静态 Spearman 拓扑图），不使用 Transformer。**
> **DynaTopo 新架构 = MSTCN + GAT（静态 Spearman 拓扑图 + 工况驱动动态图），详见下方。**
> 详见下方 [模型架构](#模型架构) 与 [版本说明](#版本说明)。

## 项目结构

```
RUL_Prediction/
├── data/                # 数据仓库（raw 原始数据 / processed 预处理数据）
├── core_models/         # 模型零件库
│   ├── stgnn_static.py       # 静态图 STGNN（原论文最终架构）
│   ├── stgnn_dynatopo.py     # 🆕 双图 STGNN（静态+工况驱动动态图）
│   ├── topo_generator/       # 🆕 动态图生成器子包（相似度/注意力）
│   └── topo_fusion/          # 🆕 图融合策略子包（特征融合/拓扑融合）
├── utils/               # 工具箱（数据处理、损失函数、评估指标）
├── configs/             # 全局配置 + dynatopo 实验配置
├── scripts/             # 训练与评估脚本
├── notebooks/           # 可视化分析与图表绘制
├── extracted_pdf/       # 课题论文 PDF 提取内容
├── saved_models/        # 训练好的模型权重
└── logs/                # 运行日志
    └── dynatopo/        # 🆕 DynaTopo 实验日志
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
| **static（原论文）** | MSTCN + GAT（Spearman固定图） | `_static` 后缀 | 论文采用的最终架构，关闭 Transformer 分支 |
| **dynatopo（新）** | MSTCN + GAT（静态图 + 工况驱动动态图） | `dynatopo_` 前缀 | 🆕 可切换 A×B 多种组合 |
| v1（已废弃） | MSTCN + GAT + Transformer | — | 已删除 |

### static 脚本（原论文复现）

- `scripts/train_basic_static.py` — 静态图单工况训练
- `scripts/ablation_static.py` — 静态图消融实验
- `scripts/evaluate_1_static.py` — 单工况评估
- `scripts/evaluate_2_static.py` — 跨工况评估
- `scripts/train_transfer.py` — 统一迁移脚本（支持 none/global_mmd/lmmd_uda/lmmd_semi）

### dynatopo 脚本（新实验）

- `scripts/train_basic_dynatopo.py --preset A1B1` — 双图模型训练（配置驱动）
- `scripts/ablation_dynatopo.py` — 双图消融实验（4种A×B + 消融对照组）
- `scripts/evaluate_1_dynatopo.py` — 双图单工况评估
- `scripts/evaluate_2_dynatopo.py` — 双图跨工况评估

### 动态图生成策略 (A)

| 策略 | 标识 | 说明 |
|------|------|------|
| A1 相似度 | `similarity` | 余弦相似度 + 工况调制 → Top-K 稀疏化 |
| A2 注意力 | `attention` | 多头注意力 + 工况偏置 → Top-K 稀疏化 |

### 图融合策略 (B)

| 策略 | 标识 | 说明 |
|------|------|------|
| B1 特征融合 | `feature` | 静态图和动态图各自过 GAT，特征层拼接融合 |
| B2 拓扑融合 | `topology` | 静态边和动态边合并去重，统一送入一个 GAT |

### 实验预设

```bash
# 4 种 A×B 组合
python scripts/train_basic_dynatopo.py --preset A1B1  # 相似度 × 特征融合
python scripts/train_basic_dynatopo.py --preset A1B2  # 相似度 × 拓扑融合
python scripts/train_basic_dynatopo.py --preset A2B1  # 注意力 × 特征融合
python scripts/train_basic_dynatopo.py --preset A2B2  # 注意力 × 拓扑融合

# 消融对照
python scripts/train_basic_dynatopo.py --preset static_only   # 仅静态图（=原STGNN）
python scripts/train_basic_dynatopo.py --preset dynamic_only  # 仅动态图

# 查看所有预设
python scripts/train_basic_dynatopo.py --list-presets
```
