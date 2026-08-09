[中文版本](./README_zh.md)

# Aero-Engine Remaining Useful Life (RUL) Prediction

This repository contains the complete code implementation for the research project **"Research on a Graph Neural Network-Based Method for Remaining Useful Life Prediction of Aero-Engines"**, using the NASA C-MAPSS dataset. It is also publicly available on GitHub for study and reference.

> **Final STGNN architecture in the paper = MSTCN + GAT (v2), without Transformer.**
> See [Model Architecture](#model-architecture) and [Version Notes](#version-notes) below.

## Project Structure

```
RUL_Prediction/
├── data/                # Data storage (raw / processed)
├── core_models/         # Model building blocks (MSTCN, GAT, Transformer, STGNN assembly)
├── utils/               # Utilities (data processing, loss functions, evaluation metrics)
├── configs/             # Global configuration
├── scripts/             # Training & evaluation scripts
├── notebooks/           # Visualization & charting
├── extracted_pdf/       # Extracted content from the research paper PDF
├── saved_models/        # Trained model weights
└── logs/                # Run logs
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
| **v2 (main)** | MSTCN + GAT | `_v2` suffix | **Final architecture used in the paper**, Transformer branch disabled |
| v1 (exploratory) | MSTCN + GAT + Transformer | no suffix | Full three-component architecture, kept for comparison |

- `scripts/train_basic_v2.py`, `scripts/evaluate_2_v2.py` — Single-condition v2 training & evaluation
- `scripts/ablation_study_v2.py` — v2 ablation study (MSTCN+GAT / MSTCN-only / GAT-only / all-off)
- `scripts/train_transfer_v2_*.py` — v2 cross-condition transfer experiments (unsupervised UDA / semi-supervised / global MMD comparison)
- Scripts without `_v2` suffix are v1 versions that include the Transformer branch
