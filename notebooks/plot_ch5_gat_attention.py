# ============================================================
# plot_ch5_gat_attention.py
# 第5章 GAT注意力权重热力图 —— 14×14 传感器间注意力强度
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader

# === Seaborn 学术风格 ===
sns.set_theme(style='whitegrid', context='paper', font_scale=1.15,
              rc={'axes.edgecolor':'0.15','grid.alpha':0.2,
                  'figure.facecolor':'white','axes.facecolor':'#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
from configs.config import (NUM_FEATURES, NUM_SENSORS, BATCH_SIZE, RANDOM_SEED,
    MSTCN_NUM_CHANNELS, MSTCN_KERNEL_SIZES, MSTCN_DROPOUT,
    GAT_HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DROPOUT, FC_HIDDEN_DIM)
from core_models.stgnn_static import STGNN_Static
from notebooks.gat_attention_helper import extract_gat_attention_matrix

np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 传感器名称
SENSOR_LABELS = ['T24','T30','T50','P30','Nf','Nc','Ps30','Phi',
                 'NRf','NRc','BPR','htBleed','W31','W32']

# 传感器类别（用于分组标注）
SENSOR_CATEGORIES = {
    '温度': ['T24','T30','T50'],
    '压力': ['P30','Ps30'],
    '转速': ['Nf','Nc','NRf','NRc'],
    '燃油': ['Phi'],
    '冷却/旁通': ['BPR','htBleed','W31','W32'],
}

print("=" * 60)
print("  第5章: GAT 注意力权重热力图")
print("=" * 60)

# === 1. 加载测试数据 ===
PROCESSED_DIR = os.path.join(PROJ_ROOT, 'data', 'processed')
test_data = np.load(os.path.join(PROCESSED_DIR, 'FD001_test.npz'))
X_test = test_data['X']  # [n_samples, 30, 17]
print(f"测试数据加载完成: {X_test.shape}")

graph = torch.load(os.path.join(PROCESSED_DIR, 'FD001_train_graph.pt'),
                   weights_only=False)
edge_index = graph['edge_index']  # [2, E]
print(f"图边数: {edge_index.shape[1]}")

# 构造 DataLoader
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_dummy = torch.zeros(len(X_test_t), 1)
test_dataset = TensorDataset(X_test_t, y_dummy)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=False)

# === 2. 加载训练好的 STGNN v2 模型 ===
MODEL_PATH = os.path.join(PROJ_ROOT, 'saved_models', 'stgnn_static_best_FD001.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型: {MODEL_PATH}。请先运行 train_basic_v2.py 训练模型。")

model = STGNN(
    num_sensors=NUM_SENSORS, num_op_settings=3,
    mstcn_channels=MSTCN_NUM_CHANNELS, mstcn_kernels=MSTCN_KERNEL_SIZES,
    mstcn_dropout=MSTCN_DROPOUT,
    gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_HEADS, gat_dropout=GAT_DROPOUT,
    trans_d_model=TRANSFORMER_D_MODEL, trans_nhead=TRANSFORMER_NHEAD,
    trans_num_layers=TRANSFORMER_NUM_LAYERS, trans_dropout=TRANSFORMER_DROPOUT,
    use_mstcn=True, use_gat=True, use_transformer=False,
    fc_hidden=FC_HIDDEN_DIM
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"模型加载完成: RMSE={checkpoint.get('best_rmse','?')}, "
      f"Score={checkpoint.get('best_score','?')}")
params = sum(p.numel() for p in model.parameters())
print(f"参数量: {params:,}")

# === 3. 提取 GAT 注意力权重 ===
print("\n提取 GAT 注意力权重...")
attn_matrix = extract_gat_attention_matrix(
    model, test_loader, edge_index, DEVICE, max_batches=20
)
print(f"注意力矩阵形状: {attn_matrix.shape}")
print(f"注意力范围: [{attn_matrix[attn_matrix>0].min():.4f}, "
      f"{attn_matrix[attn_matrix>0].max():.4f}]")

# === 4. 绘制注意力热力图 ===
fig, ax = plt.subplots(figsize=(11.5, 10))

# 使用 RdBu_r 色阶，以中位数为中心
vmax = np.percentile(attn_matrix[attn_matrix > 0], 95)
sns.heatmap(attn_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0, vmax=vmax, square=True, linewidths=0.6, linecolor='white',
            cbar_kws={'shrink':0.78, 'label':'注意力权重 α'},
            xticklabels=SENSOR_LABELS, yticklabels=SENSOR_LABELS,
            ax=ax, annot_kws={'fontsize':8.5})

ax.set_title('GAT 第一层多头注意力权重矩阵 (FD001)\n'
             '行=源传感器, 列=目标传感器, 值=聚合注意力强度',
             fontsize=14, fontweight='bold', pad=18)
ax.set_xlabel('目标传感器 (Target Node)', fontsize=12, fontweight='bold')
ax.set_ylabel('源传感器 (Source Node)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# 添加类别分色标签
# 在热力图下方添加传感器类别信息
info_text = ('颜色深度 ∝ 注意力强度  |  '
             '温度 T24 T30 T50 | 压力 P30 Ps30 | '
             '转速 Nf Nc NRf NRc | 燃油 Phi | 冷却 BPR htBleed W31 W32')
fig.text(0.5, -0.02, info_text, ha='center', fontsize=8.5, color='#7F8C8D',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f6fa',
                   edgecolor='#bdc3c7', alpha=0.9))

plt.tight_layout(pad=2.5)
out_path = os.path.join(FIG_DIR, 'ch5_gat_attention_heatmap.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\n✅ 图已保存: {out_path}")

# === 5. 分析输出：找出注意力最强的传感器对 ===
print("\n" + "=" * 60)
print("  Top-20 注意力最强的传感器对:")
print("=" * 60)
pairs = []
for i in range(NUM_SENSORS):
    for j in range(NUM_SENSORS):
        if attn_matrix[i, j] > 0:
            pairs.append((SENSOR_LABELS[i], SENSOR_LABELS[j], attn_matrix[i, j]))
pairs.sort(key=lambda x: -x[2])
for rank, (src, dst, val) in enumerate(pairs[:20], 1):
    print(f"  {rank:2d}. {src:6s} → {dst:6s}  α={val:.4f}")

# === 6. 与 Spearman 相关矩阵比较 ===
print("\n" + "=" * 60)
print("  与 Spearman 建图比较（存在边 + GAT 注意力 > 中位数）:")
print("=" * 60)
from scipy.stats import spearmanr
train_data = np.load(os.path.join(PROCESSED_DIR, 'FD001_train.npz'))
sensor_data_all = train_data['X'][:, 14, 3:]
corr_matrix, _ = spearmanr(sensor_data_all)
median_attn = np.median(attn_matrix[attn_matrix > 0])
ei = edge_index.numpy()
spearman_edges = set()
for e in range(ei.shape[1]):
    spearman_edges.add((int(ei[0,e]), int(ei[1,e])))
high_attn_pairs = []
for i in range(NUM_SENSORS):
    for j in range(NUM_SENSORS):
        if attn_matrix[i,j] > median_attn:
            high_attn_pairs.append((i,j))
overlap = sum(1 for p in high_attn_pairs if p in spearman_edges)
print(f"  Spearman 边总数: {len(spearman_edges)}")
print(f"  GAT 高注意力 (>median) 对: {len(high_attn_pairs)}")
print(f"  重叠数: {overlap}")

print("\n✅ 第5章 GAT 注意力热力图全部完成!")
