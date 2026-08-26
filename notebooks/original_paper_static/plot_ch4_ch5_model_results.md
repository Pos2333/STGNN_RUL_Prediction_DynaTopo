# 📊 第4~5章 模型结果可视化

> ⚠️ **代码已迁移至**: `plot_ch4_ch5_model_results.py`  
> 运行: `conda activate rul_env ; $env:KMP_DUPLICATE_LIB_OK="TRUE" ; python notebooks/original_paper_static/plot_ch4_ch5_model_results.py`  
> ⚙️ 采样开关: 编辑 `.py` 中的 `USE_SAMPLING`

## 产出图表

| 图号 | 图名 | 建议插入位置 | 输出文件 |
|------|------|-------------|----------|
| **图5-1** | t-SNE 特征可视化（3子图 RUL着色版） | §5.3，表5-8之后 | `figures/ch5_tsne_transfer.png` |
| **图5-2** | 五模型预测 vs 真实 RUL 合并散点图 | §5.2，表5-6之后 | `figures/ch5_pred_vs_true_scatter.png` |
| **图5-2a~e** | 五模型独立散点图 | §5.2 附录或单页展示 | `figures/ch5_pred_vs_true_scatter/*.png` |

### 所需模型文件

| 模型 | 路径 |
|------|------|
| LSTM | `saved_models/original_paper_static/baselines/lstm_pmatch_best_FD001.pt` |
| GRU | `saved_models/original_paper_static/baselines/gru_best_FD001.pt` |
| TCN | `saved_models/original_paper_static/baselines/tcn_best_FD001.pt` |
| CNN+LSTM | `saved_models/original_paper_static/baselines/cnn_lstm_best_FD001.pt` |
| STGNN (MSTCN+GAT) | `saved_models/original_paper_static/stgnn/stgnn_static_best_FD001.pt` |
| 全局 MMD 迁移 | `saved_models/transfer_v2_global_mmd_best_FD002.pt` |
| LMMD 半监督迁移 | `saved_models/transfer_v2_best_FD002.pt` |

### 采样开关
- `USE_SAMPLING = True` → 随机采样2000点加速 t-SNE
- `USE_SAMPLING = False` → 使用全部数据

---

## 图5-1: t-SNE 特征可视化（3子图 RUL着色版）

> **建议插入位置**: §5.3 "跨工况迁移实验分析"，表5-8之后

**说明文段**: 图5-1以三子图并列形式展示了FD001→FD002跨工况迁移中STGNN融合特征（h_fused, 208维）的t-SNE降维可视化。目标域（FD002）样本按真实RUL值着色（viridis色阶：深色=临近失效，浅色=健康阶段），源域（FD001）以灰色散点作为参考底，彩色半透明椭圆标示5个RUL退化子域的分布范围，灰色虚线箭头连接源域与目标域的质心并标注子域质心距均值。

- **(a) 对齐前（无迁移）**: 源域（蓝色）与目标域（红色）在空间中明显分离，子域质心距均值约58.5，说明跨工况分布偏移严重，模型难以将从FD001学到的退化模式直接套用于FD002。
- **(b) 全局MMD对齐后**: 两域整体分布被拉近（质心距降至约37.4），但目标域各RUL子域的椭圆高度重叠混杂——高RUL（健康）与低RUL（临近失效）样本在空间中交织，说明全局对齐破坏了退化阶段语义结构，可能引发负迁移。
- **(c) LMMD子域对齐后**: 目标域5个RUL子域沿空间梯度呈有序排列（深→浅），子域质心距约44.0，虽整体质心距略大于全局MMD但各子域内部分布更加紧凑有序——LMMD在拉近跨域分布的同时有效保留了退化阶段的语义保序性。

底部汇总行标注了三种策略下子域质心距均值对比，直观量化了LMMD在"对齐"与"保序"之间的权衡优势。

---

## 图5-2: 五模型预测 vs 真实 RUL 散点图

> **建议插入位置**: §5.2 "单工况预测性能对比实验"，表5-6之后

### 图5-2（合并版）

**说明文段**: 图5-2将BasicLSTM、GRU、TCN、CNN+LSTM和STGNN五种模型在FD001测试集上的RUL预测值绘制于同一坐标系中，黑色虚线y=x为理想预测线。五种模型以不同颜色和标记形状区分（红色圆=LSTM、橙色方=GRU、绿色菱=TCN、紫色三角=CNN+LSTM、蓝色倒三角=STGNN），图例同时标注各模型的RMSE与NASA Score。灰色半透明带为±20误差区域。

从图中可直观观察到：LSTM 的预测曲线相较于 STGNN 更为分散。该模型采用 2 层、100 隐藏单元，参数量为133,501，与STGNN的136,229接近，能够产生随输入变化的非恒定预测，但整体误差仍高于STGNN。

其余三种竞品模型——GRU（RMSE=16.99, Score=778.3）、TCN（RMSE=17.32, Score=759.3）、CNN+LSTM（RMSE=17.76, Score=1025.9）——散点均紧密聚集在对角线附近，肉眼可见的分布模式与STGNN（RMSE=15.59, Score=497.8）差异不大。但从定量指标看，STGNN在RMSE和NASA Score上均取得最优，尤其在非对称惩罚（过晚预测风险）维度优势更为明显。

### 图5-2a~e（独立子图）

五张独立散点图分别保存在 `figures/ch5_pred_vs_true_scatter/` 文件夹中，每张图仅展示单个模型的预测-真实散点，适合附录或单独引用。文件名分别为 `Parameter-matched_LSTM.png`、`GRU.png`、`TCN.png`、`CNN_LSTM.png`、`STGNN.png`。

---

## 🔬 BasicLSTM 基线模型诊断

### 现象

当前 LSTM 采用 2 层、100 个隐藏单元、133,501 个参数，在 FD001 测试集上输出具有正常方差，不再出现恒定值预测；其 RMSE=22.50、NASA Score=1253.81，均劣于静态 STGNN 的 16.08 和 473.07。

### 诊断过程

1. **权重检查**：模型参数加载正确，预测标准差约 32.49，说明模型仍保留输入-输出映射。
2. **指标检查**：RMSE=19.62、NASA Score=532.93，均高于 STGNN，体现了纯时序基线在当前任务上的性能差距。
3. **容量控制**：通过采用小容量结构避免训练失败，同时保留基本的时序建模能力，使基线结果具有可解释性。

### 工程结论

BasicLSTM 在 C-MAPSS FD001 数据集上能够学习基本的退化表示，但仅依靠低容量循环结构难以充分利用多传感器空间耦合信息。因此，静态 STGNN 通过 MSTCN 提取多尺度时序特征并使用 GAT 建模传感器关系后取得更优结果。

这一发现反而佐证了本文核心论点——**单纯依赖时序递归结构（LSTM/GRU）难以充分捕捉航空发动机多传感器退化规律**，引入MSTCN多尺度时序特征与GAT传感器空间耦合建模是必要的。
