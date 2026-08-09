# ============================================================
# plot_ch2_sensor_trends.py
# 第2章 传感器时序退化趋势图 —— 14传感器 × 3台发动机
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# === Seaborn 学术风格 ===
sns.set_theme(style='whitegrid', context='paper', font_scale=1.05,
              rc={'axes.edgecolor':'0.15','grid.alpha':0.2,
                  'figure.facecolor':'white','axes.facecolor':'#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
from configs.config import KEPT_SENSOR_INDICES, RANDOM_SEED
np.random.seed(RANDOM_SEED)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 14 个保留传感器名称（物理含义）
SENSOR_NAMES = ['T24','T30','T50','P30','Nf','Nc','Ps30','Phi',
                'NRf','NRc','BPR','htBleed','W31','W32']

# 传感器物理类别（用于分色）
SENSOR_CATEGORIES = {
    '温度': ['T24','T30','T50'],
    '压力': ['P30','Ps30'],
    '转速': ['Nf','Nc','NRf','NRc'],
    '燃油/控制': ['Phi'],
    '旁通/冷却': ['BPR','htBleed','W31','W32'],
}
CAT_COLORS = {'温度':'#D62728','压力':'#1F77B4','转速':'#2CA02C',
              '燃油/控制':'#FF7F0E','旁通/冷却':'#9467BD'}
name_to_color = {}
for cat, names in SENSOR_CATEGORIES.items():
    for n in names: name_to_color[n] = CAT_COLORS[cat]

# 选取的 3 台代表性发动机（FD001 训练集）
SELECTED_ENGINES = [2, 1, 91]  # 长寿命(287) / 中等(192) / 短寿命(135)
ENGINE_DESCRIPTIONS = {
    2:  f'Engine #2 (长寿命, 287 cycles)',
    1:  f'Engine #1 (中等寿命, 192 cycles)',
    91: f'Engine #91 (短寿命, 135 cycles)',
}

print("=" * 60)
print("  第2章: 传感器时序退化趋势图")
print("=" * 60)

# === 加载原始训练数据 ===
RAW_DIR = os.path.join(PROJ_ROOT, 'data', 'raw')
df = pd.read_csv(os.path.join(RAW_DIR, 'train_FD001.txt'), sep=r'\s+', header=None)
print(f"原始数据加载完成: {df.shape}")

# === 为每台发动机提取传感器时序 ===
engines_data = {}
for eng_id in SELECTED_ENGINES:
    eng_df = df[df[0] == eng_id]
    cycles = eng_df[1].values.astype(int)
    # 14 个有效传感器（0-based 原始列索引）
    sensor_data = eng_df.iloc[:, KEPT_SENSOR_INDICES].values.astype(np.float32)
    # Min-Max 归一化（每传感器独立归一化，便于跨发动机比较退化趋势）
    sensor_norm = (sensor_data - sensor_data.min(axis=0)) / \
                  (sensor_data.max(axis=0) - sensor_data.min(axis=0) + 1e-8)
    engines_data[eng_id] = {'cycles': cycles, 'sensors': sensor_norm}
    print(f"  Engine #{eng_id}: {len(cycles)} cycles, 传感器范围 [{sensor_norm.min():.3f}, {sensor_norm.max():.3f}]")

# === 绘制大图: 14 传感器 × 3 发动机 =================================
# 布局: 14 行 (传感器) × 3 列 (发动机)
N_SENSORS = 14
N_ENGINES = 3

fig, axes = plt.subplots(N_SENSORS, N_ENGINES,
                         figsize=(16, 28),
                         sharex='col', sharey='row')
fig.subplots_adjust(hspace=0.35, wspace=0.18,
                    top=0.965, bottom=0.035, left=0.08, right=0.97)

for row, s_name in enumerate(SENSOR_NAMES):
    color = name_to_color[s_name]
    for col, eng_id in enumerate(SELECTED_ENGINES):
        ax = axes[row, col]
        data = engines_data[eng_id]
        cycles = data['cycles']
        values = data['sensors'][:, row]

        ax.plot(cycles, values, color=color, linewidth=1.1, alpha=0.9)
        ax.fill_between(cycles, 0, values, color=color, alpha=0.08)

        # 标记失效点
        ax.plot(cycles[-1], values[-1], 'x', color='black', markersize=8,
                markeredgewidth=1.8, zorder=5)

        # RUL=125 截断参考线（仅在第一列显示 Y 标签）
        if col == 0:
            ax.set_ylabel(s_name, fontsize=10, fontweight='bold',
                          color=color, rotation=0, labelpad=28, va='center')

        # X 轴标签（仅在最后一行）
        if row == N_SENSORS - 1:
            ax.set_xlabel('Cycle', fontsize=9.5, fontweight='bold')

        # 列标题（仅在第一行）
        if row == 0:
            ax.set_title(ENGINE_DESCRIPTIONS[eng_id],
                         fontsize=9.5, fontweight='bold', color='#2C3E50', pad=8)

        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        # 设置 y 范围到 [0, 1.05]
        ax.set_ylim(-0.02, 1.08)

# 全局标题
fig.suptitle('FD001 传感器时序退化趋势 (3台代表性发动机)\n14个有效传感器 Min-Max归一化曲线',
             fontsize=15, fontweight='bold', y=0.992)

# 底部图例
legend_handles = [plt.Line2D([0],[0], color=c, linewidth=2.5, label=cat)
                  for cat, c in CAT_COLORS.items()]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           fontsize=9.5, framealpha=0.85, bbox_to_anchor=(0.5, -0.005))

# 底部注释
fig.text(0.5, -0.012,
         '横轴=运行周期(Cycle)　|　纵轴=归一化传感器值[0,1]　|　×=失效点　|　颜色区分传感器物理类别',
         ha='center', fontsize=8.5, color='#7F8C8D',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f6fa', edgecolor='#bdc3c7', alpha=0.85))

out_path = os.path.join(FIG_DIR, 'ch2_sensor_trends.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\n✅ 图已保存: {out_path}")

# === 额外绘制: 选4个代表性传感器做大图特写 (T30, P30, Nf, W31) ===
HIGHLIGHT_SENSORS = ['T30', 'P30', 'Nf', 'W31']
HIGHLIGHT_IDX = [SENSOR_NAMES.index(s) for s in HIGHLIGHT_SENSORS]

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for i, (s_idx, s_name) in enumerate(zip(HIGHLIGHT_IDX, HIGHLIGHT_SENSORS)):
    ax = axes2[i]
    color = name_to_color[s_name]
    for eng_id in SELECTED_ENGINES:
        data = engines_data[eng_id]
        cycles = data['cycles']
        values = data['sensors'][:, s_idx]
        label = f'Engine #{eng_id} ({len(cycles)} cycles)'
        ls = '-' if eng_id == 2 else ('--' if eng_id == 1 else ':')
        ax.plot(cycles, values, color=color, linewidth=1.6, linestyle=ls,
                alpha=0.85, label=label)
        # 失效点
        ax.plot(cycles[-1], values[-1], 'x', color='black', markersize=9,
                markeredgewidth=1.8, zorder=5)

    ax.set_title(f'{s_name}', fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel('Cycle', fontsize=10)
    ax.set_ylabel('归一化值', fontsize=10)
    ax.legend(fontsize=8.5, loc='best', framealpha=0.8)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_ylim(-0.02, 1.08)

fig2.suptitle('代表性传感器退化趋势特写 (FD001)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path2 = os.path.join(FIG_DIR, 'ch2_sensor_trends_highlight.png')
fig2.savefig(out_path2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"✅ 特写图已保存: {out_path2}")

print("\n✅ 第2章传感器退化趋势图全部完成!")
