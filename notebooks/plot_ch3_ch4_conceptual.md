# 🎨 第3~4章 概念示意图

> ⚠️ **代码已迁移至**: `plot_ch3_ch4_conceptual.py`  
> 运行: `conda activate rul_env ; $env:KMP_DUPLICATE_LIB_OK="TRUE" ; python notebooks/plot_ch3_ch4_conceptual.py`

## 产出图表

| 图号 | 图名 | 建议插入位置 | 生成方式 | 输出文件 |
|------|------|-------------|----------|----------|
| **图3-1** | MSTCN 多尺度感受野示意图 | §3.2.2 | ⭐ 生图模型 | `figures/ch3_mstcn_receptive_field.png` |
| **图3-2** | GAT 消息传递机制示意图 | §3.3.1 | ⭐ 生图模型 | `figures/ch3_gat_message_passing.png` |
| **图3-3** | 完整 STGNN 数据流图 | §3.1 | ⭐ 生图模型 | `figures/ch3_stgnn_dataflow.png` |
| **图4-1** | LMMD 子域对齐 vs 全局 MMD | §4.1.3 / §4.3.2 | ⭐ 生图模型 | `figures/ch4_lmmd_vs_mmd.png` |

> **注**: 以上 4 张概念图原本用 matplotlib 纯代码绘制，效果欠佳。现改用 AI 生图模型生成，下方提供每张图的详细 Prompt。  
> **配色要求**: Nature 期刊学术风格 —— 柔和、低饱和度、黑白打印友好。

---

## Nature 期刊学术配色参考

| 色名 | Hex | 用途 |
|------|-----|------|
| Nature Blue | `#4C72B0` | 主色调、结构框 |
| Nature Red | `#C44E52` | 强调、高亮 |
| Nature Green | `#55A868` | 正确/成功标识 |
| Nature Orange | `#DD8452` | 中间态、过渡 |
| Nature Purple | `#937860` | 辅助标记 |
| Nature Gray | `#8C8C8C` | 中性元素、背景线 |
| Nature Light | `#EAEAF2` | 背景色 |
| Nature Dark | `#2C2C2C` | 文字、边框 |

---

## 图3-1: MSTCN 多尺度时间卷积网络——时序退化特征提取

> **建议插入位置**: §3.2 "基于MSTCN的多尺度时序特征提取"，覆盖§3.2.1~§3.2.3

**说明文段**: 图3-1展示了MSTCN模块的完整内部结构。MSTCN将每个传感器的30周期时序信号视为独立的单变量序列，经三层堆叠Conv1d提取多尺度退化特征：第1层(k=3, ch=32)捕捉短期局部波动（如工况切换扰动），第2层(k=5, ch=64)感知中期退化趋势（如部件性能渐进衰退），第3层(k=7, ch=128)建模长期宏观规律（如从健康到失效的完整轨迹）。三层间使用BatchNorm+ReLU+Dropout稳定训练，全局平均池化将30周期压缩为单个128维特征向量。所有14个传感器共享同一套MSTCN权重（参数共享），最终输出[B,14,128]节点时序特征矩阵。

### 🔧 伪代码: MSTCN 前向传播

```
Algorithm: MSTCN Forward Pass (Multi-Scale Temporal Convolution)
Input:  X ∈ R^{B×N×W}       ▷ B=batch, N=14 sensors, W=30 window
Params: Conv1(1→32, k=3), Conv2(32→64, k=5), Conv3(64→128, k=7)
        BN₁, BN₂, BN₃       ▷ BatchNorm layers
Output: H ∈ R^{B×N×128}     ▷ Per-sensor temporal embeddings

 1:  // ==== Step 1: Flatten sensors as independent 1D signals ====
 2:  X ← reshape(X, [B·N, 1, W])                         ▷ [B·14, 1, 30]
 3:
 4:  // ==== Step 2: Multi-Scale Convolution Stack ====
 5:  // --- Layer 1: Short-term (k=3, receptive field=3) ---
 6:  X ← Conv1d(1→32, kernel=3, padding=1)(X)            ▷ [B·14, 32, 30]
 7:  X ← BatchNorm1d(32)(X)
 8:  X ← ReLU(X)
 9:  X ← Dropout(p=0.2)(X)
10:
11:  // --- Layer 2: Mid-term (k=5, cumulative RF expands) ---
12:  X ← Conv1d(32→64, kernel=5, padding=2)(X)           ▷ [B·14, 64, 30]
13:  X ← BatchNorm1d(64)(X)
14:  X ← ReLU(X)
15:  X ← Dropout(p=0.2)(X)
16:
17:  // --- Layer 3: Long-term (k=7, full-window context) ---
18:  X ← Conv1d(64→128, kernel=7, padding=3)(X)          ▷ [B·14, 128, 30]
19:  X ← BatchNorm1d(128)(X)
20:  X ← ReLU(X)
21:  X ← Dropout(p=0.2)(X)
22:
23:  // ==== Step 3: Temporal pooling ====
24:  X ← AdaptiveAvgPool1d(output_size=1)(X)              ▷ [B·14, 128, 1]
25:  X ← squeeze(X, dim=-1)                               ▷ [B·14, 128]
26:
27:  // ==== Step 4: Restore sensor dimension ====
28:  H ← reshape(X, [B, N, 128])                          ▷ [B, 14, 128]
29:  return H

Note: All 14 sensors share the same MSTCN weights (parameter sharing).
      Same padding ensures output length = input length = 30 throughout.
```

### 🎨 生图 Prompt

```
A comprehensive three-panel academic diagram showing the complete MSTCN (Multi-Scale Temporal Convolutional Network) architecture, in Nature journal style. Use muted academic colors: blue #4C72B0, orange #DD8452, green #55A868, gray #8C8C8C.

PANEL A (left, 25% width, titled "Input: Per-Sensor Time Series"):
- Show a 14×30 grid representing [14 sensors × 30 time cycles].
- Each row is a thin horizontal strip with varying intensity (simulated sensor readings).
- Arrow pointing right: "Reshape: [B,14,30] → [B×14, 1, 30]".
- Annotation: "14 sensors treated as independent 1D signals. Parameter sharing across all sensors."

PANEL B (center, 50% width, titled "3-Layer Multi-Scale Convolution Stack"):
- Three large vertically stacked blocks with internal detail:

  Block 1 (blue #4C72B0, "Layer 1: Short-Term (k=3, 1→32 ch)"):
  - "Conv1d(1→32, k=3, pad=1)" → "BatchNorm" → "ReLU" → "Dropout(0.2)"
  - Side note: "Captures 3-cycle local fluctuations"

  Block 2 (orange #DD8452, "Layer 2: Mid-Term (k=5, 32→64 ch)"):
  - "Conv1d(32→64, k=5, pad=2)" → "BatchNorm" → "ReLU" → "Dropout(0.2)"
  - Side note: "Captures 5-cycle degradation trends"

  Block 3 (green #55A868, "Layer 3: Long-Term (k=7, 64→128 ch)"):
  - "Conv1d(64→128, k=7, pad=3)" → "BatchNorm" → "ReLU" → "Dropout(0.2)"
  - Side note: "Captures 7-cycle macro degradation trajectory"

- Between blocks: "same padding preserves length W=30".
- Bottom shape: "[B×14, 128, 30]".

PANEL C (right, 25% width, titled "Temporal Pooling + Output"):
- "Global Average Pooling over time" with visual: 30 vertical bars compressed to single bar.
- "[B×14, 128, 1]" → "squeeze" → "[B×14, 128]"
- "Reshape → [B, 14, 128]": 14 sensor icons each with a compact 128-dim feature bar.
- Output annotation: "Per-sensor temporal embeddings for GAT input".

Overall title: "MSTCN: Multi-Scale Temporal Convolutional Network". White background, sans-serif font, thin vertical dividers between panels. All tensor shapes in italic monospace. Suitable for academic paper.

```
> **中文摘要**: 三栏学术图。左栏：14×30传感器时序网格。中栏：3层Conv1d展开(k=3→32ch/k=5→64ch/k=7→128ch)，每层BatchNorm+ReLU+Dropout，标注各层功能（短期/中期/长期）。右栏：全局平均池化+reshape→[B,14,128]。标注14传感器参数共享。Nature配色白底。

---

## 图3-2: GAT 图注意力网络——传感器空间耦合建模

> **建议插入位置**: §3.3.1 "图注意力机制原理"，公式(3-7)附近

**说明文段**: 图3-2展示了GAT模块将14个传感器建模为图节点、通过可学习注意力权重动态聚合邻域信息的完整机理。首先，MSTCN输出的节点时序特征矩阵[B,14,128]被铺平为[B×14,128]，与Spearman邻接矩阵edge_index一并输入GAT。第一层GAT使用4个注意力头并行计算注意力分数e_{ij}=LeakyReLU(aᵀ[Wh_i∥Wh_j])，经Softmax归一化为α_{ij}后加权聚合邻域特征；4头拼接后输出[B×14,256]。第二层GAT用单头融合至[B×14,64]，最终按传感器维度均值池化为全局图表示[B,64]。两跳邻居聚合使每个传感器节点能间接感知气路系统中更远的部件状态。

### 🔧 伪代码: GAT 前向传播

```
Algorithm: SensorGAT Forward Pass
Input:  X ∈ R^{B×N×D}        ▷ N=14 sensors, D=128 (MSTCN output)
        edge_index ∈ Z^{2×E}  ▷ E=124 edges from Spearman graph
Params: W¹ ∈ R^{4×128×64}     ▷ Layer1: 4 heads × (128→64)
        a¹ ∈ R^{4×256}        ▷ Layer1 attention vectors
        W² ∈ R^{256×64}        ▷ Layer2: single head fusion
        a² ∈ R^{128}           ▷ Layer2 attention vector
Output: h_graph ∈ R^{B×64}    ▷ Graph-level embedding

1:  Flatten nodes: X ← reshape(X, [B·N, D])               ▷ [B·14, 128]
2:  Repeat edge_index for each batch sample
3:
4:  // ---- Layer 1: Multi-Head Attention (4 heads, concat) ----
5:  for head h = 1 to 4 do
6:    Z_h ← X · W¹_h                                      ▷ Linear transform [B·N, 64]
7:    for each edge (i, j) ∈ edge_index do
8:      e_{ij} ← LeakyReLU(a¹_h · [Z_h_i ∥ Z_h_j])        ▷ Raw attention score
9:    for each node i do
10:     α_{ij} ← softmax_{j∈N(i)∪{i}}(e_{ij})              ▷ Normalize over neighbors
11:   H¹_h ← ELU(Σ_{j} α_{ij} · Z_h_j)                     ▷ Weighted aggregation
12: H¹ ← Concat(H¹_1, H¹_2, H¹_3, H¹_4)                   ▷ [B·N, 256]
13: H¹ ← Dropout(ReLU(H¹))
14:
15: // ---- Layer 2: Single-Head Fusion ----
16: Z² ← H¹ · W²                                          ▷ Linear [B·N, 64]
17: for each edge (i, j) do
18:   e_{ij} ← LeakyReLU(a² · [Z²_i ∥ Z²_j])
19: for each node i do
20:   α_{ij} ← softmax_{j∈N(i)∪{i}}(e_{ij})
21: H² ← ELU(Σ_{j} α_{ij} · Z²_j)                          ▷ [B·N, 64]
22:
23: // ---- Graph Readout ----
24: H² ← reshape(H², [B, N, 64])                           ▷ Recover sensor dim
25: h_graph ← mean(H², dim=1)                               ▷ Global mean pool [B, 64]
26: return h_graph
```

### 🎨 生图 Prompt

```
A comprehensive three-panel academic diagram explaining the complete Graph Attention Network (GAT) mechanism for sensor spatial coupling modeling, in Nature journal style. Use muted academic colors: blue #4C72B0, red #C44E52, orange #DD8452, green #55A868, purple #937860, gray #8C8C8C.

PANEL A (left, 30% width, titled "Input: Sensor Graph Topology"):

- Draw a complete 14-node graph with spring layout. Nodes are colored by sensor category: temperature sensors (T24,T30,T50) in red, pressure (P30,Ps30) in blue, speed (Nf,Nc,NRf,NRc) in green, fuel control (Phi) in orange, bypass/cooling (BPR,htBleed,W31,W32) in purple.
- Node labels are the sensor short names (T24, T30, etc.) in small text.
- Edges are thin gray lines, opacity proportional to |Spearman ρ|.
- Annotation: "14 sensor nodes, 62 undirected edges (|ρ|>0.6)".

PANEL B (center, 35% width, titled "Layer 1: Multi-Head Attention (H=4)"):

- Zoom into a 5-node local subgraph (nodes v0~v4 as described above).
- Show the computational flow for a single attention head (head h=1):
  a) Left side: source node features h_i, h_j (small vector bars).
  b) Linear transform W¹_h multiplies each feature → Z_h_i, Z_h_j (slightly taller bars).
  c) Concatenation [Z_h_i ∥ Z_h_j] with a bracket symbol.
  d) Dot product with attention vector a¹_h → scalar raw score e_{ij}.
  e) LeakyReLU activation symbol.
- Show this for 2 example edges: v1→v0 (strong) and v4→v0 (weak).
- Bottom: "Softmax normalization over all neighbors → α_{ij}".
- Small formula box: "e_{ij} = LeakyReLU(aᵀ · [Wh_i ∥ Wh_j])" and "α_{ij} = softmax_j(e_{ij})".

PANEL C (right, 35% width, titled "Layer 2: Fusion + Graph Readout"):

- Top: Show the 4-head concatenation from Layer 1: 4 blocks of [B·N, 64] → merged to [B·N, 256].
- Middle: Single-head GAT layer processing (simplified arrow diagram): [B·N, 256] → [B·N, 64].
- Bottom: Reshape operation: [B·N, 64] → [B, 14, 64] (grid of 14 sensor icons).
- Final: Mean pooling across sensors → graph-level vector [B, 64] (single compact bar).

Overall title at top: "GAT: Graph Attention Network for Sensor Spatial Coupling". Each panel separated by subtle vertical dividers. White background, clean sans-serif font, academic vector style. Annotation at bottom: "N(i) = neighbors of node i from Spearman correlation graph (edge_index)".
```
---

## 图3-3: 完整 STGNN 时空图神经网络架构

> **建议插入位置**: §3.1 "总体网络架构设计"，作为该节总结图

**说明文段**: 图3-3展示了STGNN模型的完整端到端架构，包含MSTCN时序编码、GAT空间建模和特征融合三个核心阶段。输入滑动窗口[B,30,17]按通道拆分为操作参数(前3维)和传感器数据(后14维)。操作参数经Conv1d(k=3)+全局池化压缩为[B,16]工况编码。传感器分支：首先permute为[B,14,30]输入MSTCN（3层堆叠Conv1d, k=3→5→7, ch=32→64→128, 参数共享），输出[B,14,128]每传感器时序特征；随后铺平为[B×14,128]，结合Spearman邻接矩阵(124条边)送入2层GAT（Layer1: 4头拼接→256维, Layer2: 单头融合→64维），经传感器维度均值池化得[B,64]空间特征。三路特征（工况16 + GAT空间64 + 全局时序128）= [B,208]拼接后，经FC(208→64→1)+ReLU输出RUL预测值ŷ∈[0,125]。

### 🔧 伪代码: STGNN 完整前向传播

```
Algorithm: STGNN Full Forward Pass (v2: MSTCN + GAT)
Input:  X ∈ R^{B×30×17}      ▷ Sliding window samples
        edge_index ∈ Z^{2×E}  ▷ Spearman graph (E=124)
Output: ŷ ∈ R^{B×1}           ▷ Predicted RUL ∈ [0,125]

1:  // ==== Split input channels ====
2:  op ← X[:, :, :3]                                    ▷ [B, 30, 3]  operating params
3:  s  ← X[:, :, 3:]                                    ▷ [B, 30, 14] sensor readings
4:
5:  // ==== Branch 1: Operating Condition Encoding ====
6:  op ← permute(op, (0,2,1))                           ▷ [B, 3, 30]
7:  op ← Conv1d(3→16, k=3) → BN → ReLU                  ▷ [B, 16, 30]
8:  op ← AdaptiveAvgPool1d(op) → squeeze                 ▷ [B, 16]
9:
10: // ==== Branch 2: MSTCN Multi-Scale Temporal ====
11: s ← permute(s, (0,2,1))                              ▷ [B, 14, 30]
12: h_temporal ← MSTCN(s)                                 ▷ [B, 14, 128]
13: //    MSTCN detail: reshape [B·14, 1, 30]
14: //    Conv1d(1→32, k=3) → BN → ReLU → Dropout       ▷ short-term
15: //    Conv1d(32→64, k=5) → BN → ReLU → Dropout      ▷ mid-term
16: //    Conv1d(64→128, k=7) → BN → ReLU → Dropout     ▷ long-term
17: //    GlobalAvgPool1d → [B·14, 128] → reshape [B, 14, 128]
18:
19: // ==== Branch 3: GAT Spatial Coupling ====
20: h_nodes ← reshape(h_temporal, [B·14, 128])           ▷ Flatten sensors
21: ei_batch ← repeat_edge_index(edge_index, B)           ▷ Offset for batching
22:
23: // GAT Layer 1: 4-head attention (concat)
24: h_nodes ← GATConv(128→64, heads=4, concat=True)      ▷ [B·14, 256]
25: h_nodes ← ReLU → Dropout
26: // GAT Layer 2: single-head fusion
27: h_nodes ← GATConv(256→64, heads=1, concat=False)     ▷ [B·14, 64]
28: h_nodes ← ReLU
29:
30: h_spatial ← reshape(h_nodes, [B, 14, 64])            ▷ Recover sensors
31: h_spatial ← mean(h_spatial, dim=1)                    ▷ [B, 64] graph readout
32:
33: // ==== Branch 4: Global Temporal Context ====
34: h_global ← mean(s, dim=1)                             ▷ [B, 14] time average
35: h_global ← Linear(14→28) → ReLU → Linear(28→128)     ▷ [B, 128]
36:
37: // ==== Feature Fusion & Prediction ====
38: h_fused ← Concat(op, h_spatial, h_global)             ▷ [B, 16+64+128] = [B, 208]
39: h_fused ← Dropout(h_fused, p=0.3)
40: ŷ ← Linear(208→64) → ReLU → Dropout → Linear(64→1)  ▷ [B, 1]
41: ŷ ← ReLU(ŷ)                                           ▷ RUL ≥ 0 constraint
42: return ŷ
```

### 🎨 生图 Prompt

```
A comprehensive vertical flow diagram showing the complete STGNN (Spatio-Temporal Graph Neural Network) architecture with detailed internal module structures, in Nature journal style. Use muted academic colors: dark gray #2C2C2C, blue #4C72B0, red #C44E52, orange #DD8452, green #55A868, purple #937860, teal #17BECF.

LAYOUT: Vertical pipeline from top to bottom, with three main horizontal sections separated by dashed divider lines.

SECTION 1 (top, light gray background #F5F5F5, titled "Input & Channel Split"):
- Large rounded box labeled "Sliding Window Input [B, 30, 17]" with annotation "B=batch, 30=cycles, 17=(3 op + 14 sensors)".
- Downward split into two streams:
  - Left stream: "Op Settings [B, 30, 3]" → "permute(0,2,1)" → "Conv1d(k=3, 3→16)" → "BN+ReLU" → "AvgPool" → small box "[B, 16]" (colored teal #17BECF).
  - Right stream: "Sensor Readings [B, 30, 14]" branching into two sub-streams.

SECTION 2 (middle, titled "MSTCN: Multi-Scale Temporal Feature Extraction"):
- Zoom-in box showing MSTCN internals:
  a) "Reshape: [B,14,30] → [B×14, 1, 30]" (treat each sensor as independent 1D signal).
  b) Three stacked Conv1d layers, each shown as a rounded block with internal detail:
     Layer 1 (blue #4C72B0): "Conv1d(1→32, k=3, pad=1)" → "BN" → "ReLU" → "Dropout(0.2)" — annotation: "Captures short-term fluctuations (3-cycle window)".
     Layer 2 (orange #DD8452): "Conv1d(32→64, k=5, pad=2)" → "BN" → "ReLU" → "Dropout(0.2)" — annotation: "Captures mid-term degradation (5-cycle window)".
     Layer 3 (green #55A868): "Conv1d(64→128, k=7, pad=3)" → "BN" → "ReLU" → "Dropout(0.2)" — annotation: "Captures long-term macro trends (7-cycle window)".
  c) All 14 sensors share the same MSTCN weights (annotation: "Parameter Sharing across all sensors").
  d) Output: "Global Average Pooling over time axis" → "[B×14, 128]" → "Reshape → [B, 14, 128]".

SECTION 3 (bottom, titled "GAT: Graph Attention Spatial Coupling + Fusion"):
- Left side: Small icon of Spearman correlation graph with 14 nodes and edge connections, arrow pointing to "edge_index [2, 124]".
- Input: "Flatten: [B,14,128] → [B×14, 128]" + edge_index.
- GAT Layer 1 block (purple #937860): "GATConv(128→64, heads=4, concat)" with sub-annotation: "Multi-head attention: e_ij = LeakyReLU(aᵀ[Wh_i∥Wh_j])" → output "[B×14, 256]".
- ReLU + Dropout.
- GAT Layer 2 block (purple): "GATConv(256→64, heads=1, concat=False)" → output "[B×14, 64]".
- "Reshape → [B, 14, 64]" → "Mean Pool over sensors" → output box "[B, 64]" (colored purple).

FUSION SECTION:
- Three incoming arrows labeled with their shapes:
  - From Section 1 left: "Op Encoding [B, 16]" (teal).
  - From Section 3: "GAT Spatial [B, 64]" (purple).
  - From Section 2 (bypass): "Global Temporal [B, 128]" (green, via time-dimension mean pool + Linear).
- Merge into large box (blue #4C72B0): "Feature Concatenation → [B, 16+64+128] = [B, 208]".

PREDICTION HEAD:
- "Dropout(p=0.3)" → "Linear(208→64) → ReLU" → "Dropout" → "Linear(64→1) → ReLU".
- Final output box (red #C44E52): "RUL Prediction ŷ ∈ [0, 125]".

Legend at top-right: color-coded module types (Input=gray, Temporal=orange, Spatial=purple, Fusion=blue, Output=red). Overall title: "STGNN: Spatio-Temporal Graph Neural Network for RUL Prediction". White background, clean rounded rectangles with 1px borders, sans-serif 9-11pt font. All tensor shapes in italic monospace. Suitable for full-page academic paper figure.
```

> **中文摘要**: 三区纵向架构图。顶部：输入拆分。中部：MSTCN展开为3层Conv1d(k=3/5/7, ch=32/64/128)，标注每层功能（短期/中期/长期），14传感器参数共享。底部：GAT展开为2层（4头→256维，单头→64维）+ 均值池化。融合区：三路拼接[B,208]→FC→RUL输出。Nature配色，白底学术风格。

---

## 图4-1: LMMD 子域对齐 vs 全局 MMD 对比

> **建议插入位置**: §4.1.3 "从全局分布对齐到子域分布对齐的必要性" 或 §4.3.2 "全局MMD的局限性"

**说明文段**: 图4-1以概念椭圆对比了全局MMD与LMMD子域对齐两种策略。左图（全局MMD）：5个不同颜色的退化子域椭圆（红→橙→黄→绿→蓝，对应RUL从低到高），源域和目标域各一组，虚线箭头错误地将不同颜色的子域拉近（如将健康阶段蓝色椭圆与临近失效阶段红色椭圆匹配），标注"退化阶段混淆"。右图（LMMD）：相同颜色编码，但实线箭头仅在相同颜色（同一退化阶段）内进行局部分布匹配，标注"退化阶段匹配"。顶部总标题，底部图例说明椭圆表示特征分布云。此图为概念示意图，非真实数据图。

### 🔧 伪代码: MMD 基础距离度量

```
Algorithm: Maximum Mean Discrepancy (MMD)
Input:  Z_s ∈ R^{N_s×D}     ▷ Source domain features (e.g., fused_feat)
        Z_t ∈ R^{N_t×D}     ▷ Target domain features
Params: kernel_mul=2.0, kernel_num=5  ▷ Multi-bandwidth RBF kernel
Output: mmd ∈ R              ▷ Distribution discrepancy (lower = more similar)

 1:  // ==== Step 1: Compute pairwise L2 distance matrix ====
 2:  Z ← Concat(Z_s, Z_t)                               ▷ [N_s+N_t, D]
 3:  L2² ← ||Z_i - Z_j||²  for all i,j                   ▷ [N_s+N_t, N_s+N_t]
 4:
 5:  // ==== Step 2: Multi-bandwidth Gaussian kernel matrix ====
 6:  σ²_base ← Σ(L2²) / ((N_s+N_t)² - (N_s+N_t))        ▷ Median distance heuristic
 7:  K ← zeros(N_s+N_t, N_s+N_t)
 8:  for m = 0 to kernel_num-1 do
 9:    σ²_m ← σ²_base × (kernel_mul)^m                    ▷ Scaled bandwidth
10:    K ← K + exp(-L2² / (σ²_m + ε))                     ▷ Sum of Gaussian kernels
11:
12:  // ==== Step 3: MMD² = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)] ====
13:  K_ss ← K[0:N_s, 0:N_s]                              ▷ Source-source block
14:  K_tt ← K[N_s:, N_s:]                                 ▷ Target-target block
15:  K_st ← K[0:N_s, N_s:]                                ▷ Source-target block
16:  mmd ← mean(K_ss) + mean(K_tt) - mean(K_st) - mean(K_st^T)
17:  return max(mmd, 0)                                   ▷ Ensure non-negative
```

### 🔧 伪代码: LMMD 局部最大均值差异

```
Algorithm: Local Maximum Mean Discrepancy (LMMD)
Input:  feat_s ∈ R^{N_s×D}   ▷ Source fused features
        feat_t ∈ R^{N_t×D}   ▷ Target fused features
        y_s ∈ R^{N_s}         ▷ Source RUL labels (for subdomain partitioning)
Params: C = 5                 ▷ Number of subdomains
Output: lmmd ∈ R              ▷ Averaged subdomain-level MMD

 1:  // ==== Step 1: Quantile-based subdomain boundaries ====
 2:  y_sorted ← sort(y_s)                                ▷ Sort RUL labels ascending
 3:  boundaries ← []                                      ▷ C-1 boundary values
 4:  for c = 1 to C-1 do
 5:    idx ← floor(N_s × c / C)
 6:    boundaries.append(y_sorted[idx])
 7:
 8:  // ==== Step 2: Per-subdomain MMD computation ====
 9:  mmd_sum ← 0.0; valid ← 0
10:  for c = 0 to C-1 do
11:    // Determine subdomain range in source labels
12:    if c == 0:
13:      mask ← (y_s ≤ boundaries[0])                     ▷ Subdomain 1: lowest RUL
14:    elif c == C-1:
15:      mask ← (y_s > boundaries[C-2])                   ▷ Subdomain C: highest RUL
16:    else:
17:      mask ← (y_s > boundaries[c-1]) ∧ (y_s ≤ boundaries[c])
18:
19:    // Skip if too few samples (avoid unstable kernel estimation)
20:    if sum(mask) < 3: continue
21:
22:    feat_sub ← feat_s[mask]                             ▷ Source features in subdomain c
23:    mmd_sum ← mmd_sum + MMD(feat_sub, feat_t)           ▷ Align subdomain c ↔ all target
24:    valid ← valid + 1
25:
26:  if valid == 0: return 0.0
27:  return mmd_sum / valid                                ▷ Average over valid subdomains
```

### 🔧 伪代码: LMMD 迁移学习训练循环

```
Algorithm: Transfer Learning Training (UDA + Semi-Supervised variants)
Input:  Pretrained model M₀ (from FD001)
        D_s = {(X_sⁱ, y_sⁱ)}  ▷ Source domain (FD001, labeled)
        D_t = {(X_tʲ, y_tʲ)}  ▷ Target domain (FD002/3/4, labels optional)
        edge_s, edge_t        ▷ Sensor graph structures
Params: λ = 0.1               ▷ LMMD loss weight
        w = 0.0 or 1.0        ▷ Target task weight (0=UDA, 1=Semi-Sup)
Output: Fine-tuned model M

 1:  M ← M₀                                             ▷ Initialize from FD001 pretrained
 2:  opt ← Adam(M.parameters, lr=0.001)
 3:  sched ← ReduceLROnPlateau(opt, patience=5, factor=0.5)
 4:  best_loss ← ∞; patience_cnt ← 0
 5:
 6:  for epoch = 1 to N_EPOCHS do
 7:    // ==== Training Phase ====
 8:    M.train()
 9:    for each mini-batch (X_s_b, y_s_b), (X_t_b, y_t_b) do
10:      // Forward pass — extract predictions and fused features
11:      ŷ_s, feat_s ← M(X_s_b, edge_s, return_feat=True)  ▷ Source predictions + features
12:      ŷ_t, feat_t ← M(X_t_b, edge_t, return_feat=True)  ▷ Target predictions + features
13:
14:      // Task loss
15:      L_task ← 0.5·MSE(ŷ_s, y_s) + 0.5·Score(ŷ_s, y_s)  ▷ Source always supervised
16:      if w > 0:                                          ▷ Semi-supervised variant
17:        L_task ← L_task + w·[0.5·MSE(ŷ_t, y_t) + 0.5·Score(ŷ_t, y_t)]
18:
19:      // LMMD alignment loss (bidirectional)
20:      L_lmmd_s2t ← LMMD(feat_s, feat_t, y_s)              ▷ Source subdomains → target
21:      L_lmmd_t2s ← LMMD(feat_t, feat_s, y_t)              ▷ Target subdomains → source
22:      L_lmmd ← (L_lmmd_s2t + L_lmmd_t2s) / 2.0
23:
24:      // Total loss
25:      L_total ← L_task + λ · L_lmmd
26:
27:      // Backward
28:      opt.zero_grad()
29:      L_total.backward()
30:      clip_grad_norm(M.parameters, max_norm=1.0)          ▷ Prevent gradient explosion
31:      opt.step()
32:
33:    // ==== Validation Phase ====
34:    M.eval()
35:    L_val ← evaluate_task_loss(M, D_t_val, edge_t)        ▷ Target domain validation
36:    sched.step(L_val)
37:
38:    // ==== Early Stopping ====
39:    if L_val < best_loss:
40:      best_loss ← L_val; patience_cnt ← 0
41:      save_checkpoint(M)                                  ▷ Save best model
42:    else:
43:      patience_cnt ← patience_cnt + 1
44:      if patience_cnt ≥ EARLY_STOP_PATIENCE: break
45:
46:  return M  ▷ Fine-tuned model adapted to target domain

Key distinction:
  - UDA (w=0):     Target labels NOT used for task loss; alignment purely via LMMD.
  - Semi-Sup (w=1): Target labels ALSO used for task loss (equal weight to source).
  - Bidirectional LMMD: Both s→t and t→s alignments prevent asymmetric drift.
  - Gradient clipping: Essential due to combined MSE + Score + LMMD gradients.
```

### 🎨 生图 Prompt

```
A two-panel conceptual diagram comparing Global MMD versus LMMD subdomain alignment strategies for domain adaptation, in Nature journal style. Use smooth semi-transparent ellipses to represent feature distribution clouds (NOT scatter points). 

Common setup for both panels: 5 degradation subdomains, each with a distinct muted color from warm to cool representing RUL from low to high:
- Subdomain 1 (near failure, RUL 0-25): muted red #C44E52
- Subdomain 2 (severe degradation, RUL 25-50): muted orange #DD8452  
- Subdomain 3 (moderate degradation, RUL 50-75): muted yellow-green #C4A43E
- Subdomain 4 (mild degradation, RUL 75-100): muted green #55A868
- Subdomain 5 (healthy, RUL 100-125): muted blue #4C72B0

Source domain (FD001) ellipses positioned on the left side, target domain (FD002-FD004) ellipses shifted to the right, creating a visible domain gap. Both domains contain all 5 colored subdomain ellipses in the same vertical order (healthy at top, near-failure at bottom).

Panel A (left, titled "Global MMD — Degradation Stage Confusion", with red accent):
- A large dashed outline ellipse covering all source subdomains and another covering all target subdomains, representing global alignment scope.
- Gray dashed arrows crossing between DIFFERENT-colored ellipses (e.g., source healthy-blue → target near-failure-red), illustrating incorrect cross-stage matching.
- Annotation box: "Global MMD aligns without stage awareness — different degradation stages get mixed up".

Panel B (right, titled "LMMD — Degradation Stage Preservation", with green accent):
- Solid colored arrows connecting ONLY same-colored ellipses between source and target (e.g., source red → target red, source blue → target blue), illustrating correct within-stage matching.
- Five small label boxes near each arrow pair naming the subdomain (e.g., "Near Failure [0,25]", "Healthy [100,125]").
- Annotation box: "LMMD aligns within each degradation stage — preserves RUL semantic structure".

Overall title: "Distribution Alignment Strategy: Global MMD vs LMMD Subdomain Alignment". Bottom legend: "Ellipses represent feature distribution clouds of each degradation stage. Arrows indicate alignment direction." White background, clean sans-serif font, suitable for academic publication. No scatter points — conceptual ellipses only.
```

> **中文摘要**: 双栏概念对比图，用半透明椭圆表示分布云（非散点）。左栏"全局MMD"：灰色虚线箭头跨颜色连接（健康蓝→失效红），标注"退化阶段混淆"。右栏"LMMD"：实线箭头仅在同色椭圆间连接，标注"退化阶段匹配"。5个子域颜色从红(失效)到蓝(健康)，左右两域各有5个椭圆。白底学术风格，明确标注"概念示意图"。
