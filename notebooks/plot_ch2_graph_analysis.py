# ============================================================
# plot_ch2_graph_analysis.py
# 第2章 数据预处理可视化 —— Spearman 热力图 & 传感器拓扑图
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import torch, networkx as nx, seaborn as sns

# === Seaborn 学术风格 ===
sns.set_theme(style='whitegrid', context='paper', font_scale=1.15,
              rc={'axes.edgecolor':'0.15','grid.alpha':0.25,
                  'figure.facecolor':'white','axes.facecolor':'#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
from configs.config import NUM_SENSORS, GRAPH_THRESHOLD, RANDOM_SEED, KEPT_SENSOR_INDICES
np.random.seed(RANDOM_SEED)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SENSOR_LABELS = ['T24','T30','T50','P30','Nf','Nc','Ps30','Phi','NRf','NRc','BPR','htBleed','W31','W32']
SENSOR_CATEGORIES = {
    '温度传感器':       ['T24','T30','T50'],
    '压力传感器':       ['P30','Ps30'],
    '转速传感器':       ['Nf','Nc','NRf','NRc'],
    '燃油/控制传感器':  ['Phi'],
    '旁通/冷却传感器':  ['BPR','htBleed','W31','W32'],
}

# === 加载数据（与 data_processor.py 一致：使用全部原始数据） ===
RAW_DIR = os.path.join(PROJ_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJ_ROOT, 'data', 'processed')

# 加载原始数据（与 data_processor.build_graph_structure 完全一致）
# data_processor 使用全部原始样本（20,631行），而非窗口化后的单时间切片
df = pd.read_csv(os.path.join(RAW_DIR, 'train_FD001.txt'), sep=r'\s+', header=None)
sensor_data_all = df.iloc[:, KEPT_SENSOR_INDICES].values.astype(np.float32)
print(f"数据就绪: {sensor_data_all.shape} (全部原始数据, 与 data_processor 一致)")

# ======================== 图1: Spearman 热力图 ========================
print("计算 Spearman 相关矩阵...")
corr_matrix, _ = spearmanr(sensor_data_all)
strong_pairs = np.sum(np.abs(corr_matrix) > GRAPH_THRESHOLD) - NUM_SENSORS

fig, ax = plt.subplots(figsize=(11.5, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.6, linecolor='white',
            cbar_kws={'shrink':0.78, 'label':"Spearman's ρ"},
            xticklabels=SENSOR_LABELS, yticklabels=SENSOR_LABELS,
            ax=ax, annot_kws={'fontsize':8.5})
ax.set_title('传感器 Spearman 秩相关系数矩阵 (FD001)', fontsize=16, fontweight='bold', pad=18)
ax.set_xlabel('传感器名称', fontsize=13, fontweight='bold')
ax.set_ylabel('传感器名称', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(rotation=0, fontsize=10)
info_text = f'建边阈值 |ρ| > {GRAPH_THRESHOLD}　　强相关传感器对: {strong_pairs//2} 对'
fig.text(0.5, -0.02, info_text, ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(FIG_DIR, 'ch2_spearman_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("图1 已保存: ch2_spearman_heatmap.png")

# ======================== 图2: 传感器拓扑图 ========================
print("构建传感器拓扑图...")
edge_index = torch.load(os.path.join(PROCESSED_DIR, 'FD001_train_graph.pt'), weights_only=False)['edge_index']
G = nx.Graph(); G.add_nodes_from(range(NUM_SENSORS))
for i in range(edge_index.shape[1]):
    src, dst = int(edge_index[0,i]), int(edge_index[1,i])
    if src < dst: G.add_edge(src, dst, weight=abs(corr_matrix[src, dst]))
print(f"节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}, 密度: {nx.density(G):.3f}")

cat_colors = {'温度传感器':'#D62728','压力传感器':'#1F77B4','转速传感器':'#2CA02C',
              '燃油/控制传感器':'#FF7F0E','旁通/冷却传感器':'#9467BD'}
node_to_cat = {}
for cat, labels in SENSOR_CATEGORIES.items():
    for l in labels: node_to_cat[SENSOR_LABELS.index(l)] = cat
node_colors = [cat_colors[node_to_cat[n]] for n in G.nodes()]
degrees = dict(G.degree()); max_deg = max(degrees.values())
node_sizes = [380 + 130 * degrees[n]/max_deg for n in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=2.8, iterations=150)

fig, ax = plt.subplots(figsize=(13, 10.5))
edge_weights = [G[u][v]['weight']*2.2 for u,v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.28, edge_color='#7f8c8d', ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       alpha=0.9, edgecolors='white', linewidths=2.2, ax=ax)
label_pos = {k:(x,y+0.055) for k,(x,y) in pos.items()}
nx.draw_networkx_labels(G, label_pos, labels={i:SENSOR_LABELS[i] for i in range(NUM_SENSORS)},
                        font_size=9.5, font_weight='bold', font_color='#2c3e50', ax=ax)
legend_handles = [mpatches.Patch(color=c, alpha=0.9, label=n) for n,c in cat_colors.items()]
ax.legend(handles=legend_handles, loc='lower left', fontsize=9.5, framealpha=0.85,
          title='传感器类别', title_fontsize=10.5, ncol=2)
ax.set_title('传感器空间拓扑图 (FD001)\nSpearman |ρ| > 0.6', fontsize=16, fontweight='bold', pad=18)
stats_text = (f'节点数: {G.number_of_nodes()}  |  边数: {G.number_of_edges()}  |  '
              f'平均度数: {2*G.number_of_edges()/G.number_of_nodes():.1f}  |  图密度: {nx.density(G):.3f}')
ax.text(0.5, -0.03, stats_text, transform=ax.transAxes, fontsize=9.5, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f6fa', edgecolor='#bdc3c7', alpha=0.9))
ax.axis('off'); plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'ch2_sensor_topology.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("图2 已保存: ch2_sensor_topology.png")

print("\n节点度数统计:")
for i in range(NUM_SENSORS):
    print(f"  {SENSOR_LABELS[i]:8s} ({node_to_cat[i]:12s}): 度数={degrees[i]}")
print("\n✅ 第2章图表全部完成!")
