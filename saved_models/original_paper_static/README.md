# 原论文静态模型权重

本目录收纳原论文静态图 STGNN 及竞品模型相关权重。

## 目录结构

- `baselines/`：Parameter-matched LSTM、GRU、TCN、CNN+LSTM 的正式权重。
- `stgnn/`：静态 STGNN 最佳权重和 checkpoint。
- `stgnn/archive/`：2026-08-16 的静态 STGNN 历史备份。
- `ablation/`：静态 STGNN 消融实验输出目录（当前无正式权重）。
- `transfer/none/`：静态 STGNN 无自适应目标域微调权重。
- `transfer/lmmd_semi/`：静态 STGNN 半监督 LMMD 权重。
- `transfer/lmmd_uda/`：静态 STGNN 无监督 LMMD/UDA 权重。

动态图与动态图迁移权重仍保留在 `saved_models/` 根目录，不属于本目录。

当前唯一正式 LSTM 为 Parameter-matched LSTM（2 层、100 隐藏单元、133,501 参数），权重文件为 `baselines/lstm_pmatch_best_FD001.pt`。
