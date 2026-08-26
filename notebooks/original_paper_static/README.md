# 原论文静态 STGNN 资产

本目录收纳原论文中静态图 STGNN、竞品模型和数据分析相关的 Notebook、绘图脚本、说明文档及生成图表。

## 目录内容

- `data_exploration.ipynb`：FD001 数据探索与特征筛选。
- `plot_ch2_*.py/.md`：第 2 章数据、RUL 标签、滑动窗口及静态拓扑图。
- `plot_ch3_ch4_conceptual.py/.md`：MSTCN、GAT、静态 STGNN 和 LMMD 概念图。
- `plot_ch4_ch5_model_results.py/.md`：LSTM、GRU、TCN、CNN+LSTM、静态 STGNN 五模型对比图（LSTM 为参数匹配结构）。
- `plot_ch5_gat_attention.py/.md`、`gat_attention_helper.py`：静态 GAT 注意力可解释性。
- `figures/`：上述脚本输出的原论文图表。

## 运行

所有命令均从项目根目录执行，例如：

`python notebooks/original_paper_static/plot_ch4_ch5_model_results.py`

五模型权重位于 `saved_models/original_paper_static/`。

## 当前五模型结果

| 模型 | RMSE | NASA Score |
|---|---:|---:|
| LSTM | 22.50 | 1253.81 |
| STGNN | **16.08** | **473.07** |
| GRU | 16.99 | 778.26 |
| TCN | 17.32 | 759.26 |
| CNN+LSTM | 17.76 | 1025.90 |

`ch5_tsne_transfer.png` 为历史静态迁移产物；当前绘图脚本不重新生成它，因为旧版全局 MMD 权重缺失。
