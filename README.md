[中文版本](./README_zh.md)

# Aero-Engine Remaining Useful Life (RUL) Prediction

This repository contains the complete code implementation for the research project **"Research on a Graph Neural Network-Based Method for Remaining Useful Life Prediction of Aero-Engines"**, using the NASA C-MAPSS dataset. It is also publicly available on GitHub for study and reference.

> **Paper STGNN final architecture = MSTCN + GAT (static Spearman topology), without Transformer.**
> **DynaTopo new architecture = MSTCN + GAT (static Spearman topology + condition-driven dynamic graph), see below.**
> See [Model Architecture](#model-architecture) and [Version Notes](#version-notes) below.

## Project Structure

```
RUL_Prediction/
├── data/                # Data storage (raw / processed)
├── core_models/         # Model building blocks
│   ├── stgnn_static.py       # Static-graph STGNN (original paper architecture)
│   ├── stgnn_dynatopo.py     # 🆕 Dual-graph STGNN (static + condition-driven dynamic)
│   ├── topo_generator/       # 🆕 Dynamic graph generator subpackage (similarity/attention)
│   └── topo_fusion/          # 🆕 Graph fusion strategy subpackage (feature/topology)
├── utils/               # Utilities (data processing, loss functions, evaluation metrics)
├── configs/             # Global config + dynatopo experiment config
├── scripts/             # Training & evaluation scripts
├── notebooks/           # Visualization & charting
├── extracted_pdf/       # Extracted content from the research paper PDF
├── saved_models/        # Trained model weights
└── logs/                # Run logs
    └── dynatopo/        # 🆕 DynaTopo experiment logs
```

## Environment Setup

```bash
# Create virtual environment
conda create -n rul_env python=3.10 -y
conda activate rul_env

# Install dependencies
pip install -r requirements.txt
```

## Dataset

NASA C-MAPSS dataset (FD001 ~ FD004), located under `data/raw/`.

## Model Architecture

The final **STGNN (Spatio-Temporal Graph Neural Network)** adopted in the paper consists of the following modules:

- **MSTCN** (Multi-Scale Temporal Convolutional Network): Three stacked Conv1d layers (kernel 3→5→7, channels 32→64→128) with parameter sharing, extracting multi-scale temporal degradation features across 14 sensors
- **GAT** (Graph Attention Network): Two-layer multi-head attention (4-head → 1-head), modeling spatial coupling relationships over a sensor topology graph built via Spearman rank correlation
- **Global Temporal Context**: Time-dimension mean pooling + two Linear layers, providing supplementary global degradation trend information
- **Operating Condition Encoding**: Conv1d encoding of 3 operating parameters
- **Feature Fusion**: Concatenation of operating condition encoding, GAT spatial features, and global temporal context → FC → RUL
- **Loss Function**: MSE + NASA asymmetric scoring joint loss
- **Transfer Learning**: LMMD (Local Maximum Mean Discrepancy) cross-condition adaptation based on RUL degradation stage subdomain partitioning

> Note: The `Transformer` module (`core_models/transformer.py`) and its ablation switch `use_transformer` are retained in the codebase for architecture exploration and comparative experiments. Ablation studies showed that the Transformer branch contributes limitedly to final performance, so the paper's final STGNN model **does not use Transformer**.

## Version Notes

| Version | Architecture | Identifier | Description |
|---------|-------------|------------|-------------|
| **static (paper)** | MSTCN + GAT (Spearman fixed graph) | `_static` suffix | Final architecture in the paper, Transformer disabled |
| **dynatopo (new)** | MSTCN + GAT (static + condition-driven dynamic) | `dynatopo_` prefix | 🆕 Switchable A×B combinations |
| v1 (deprecated) | MSTCN + GAT + Transformer | — | Deleted |

### Static Scripts (paper reproduction)

- `scripts/train_basic_static.py` — Static-graph single-condition training
- `scripts/ablation_static.py` — Static-graph ablation study
- `scripts/evaluate_1_static.py` — Single-condition evaluation
- `scripts/evaluate_2_static.py` — Cross-condition evaluation
- `scripts/train_transfer.py` — Unified transfer script (supports none/global_mmd/lmmd_uda/lmmd_semi)

### DynaTopo Scripts (new experiments)

- `scripts/train_basic_dynatopo.py --preset A1B1` — Dual-graph training (config-driven)
- `scripts/ablation_dynatopo.py` — Dual-graph ablation study (4 A×B + ablation controls)
- `scripts/evaluate_1_dynatopo.py` — Dual-graph single-condition evaluation
- `scripts/evaluate_2_dynatopo.py` — Dual-graph cross-condition evaluation

### Dynamic Graph Generation Strategies (A)

| Strategy | ID | Description |
|----------|------|-------------|
| A1 Similarity | `similarity` | Cosine similarity + condition modulation → Top-K sparsification |
| A2 Attention | `attention` | Multi-head attention + condition bias → Top-K sparsification |

### Graph Fusion Strategies (B)

| Strategy | ID | Description |
|----------|------|-------------|
| B1 Feature Fusion | `feature` | Static and dynamic graphs each through independent GATs, feature-level concatenation |
| B2 Topology Fusion | `topology` | Merge and deduplicate static & dynamic edges, unified single GAT |

### Experiment Presets

```bash
# 4 A×B combinations
python scripts/train_basic_dynatopo.py --preset A1B1  # Similarity × Feature Fusion
python scripts/train_basic_dynatopo.py --preset A1B2  # Similarity × Topology Fusion
python scripts/train_basic_dynatopo.py --preset A2B1  # Attention × Feature Fusion
python scripts/train_basic_dynatopo.py --preset A2B2  # Attention × Topology Fusion

# Ablation controls
python scripts/train_basic_dynatopo.py --preset static_only   # Static only (= original STGNN)
python scripts/train_basic_dynatopo.py --preset dynamic_only  # Dynamic only

# List all presets
python scripts/train_basic_dynatopo.py --list-presets
```
