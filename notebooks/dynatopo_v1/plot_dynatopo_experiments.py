# ============================================================
# plot_dynatopo_experiments.py
# 动态拓扑实验核心图表集合
#
# 产出:
#   figures/dynatopo_multiseed_boxplot.png  — 多seed稳定性箱线图
#   figures/dynatopo_strategy_heatmap.png   — 2×2策略矩阵性能热力图
#   figures/dynatopo_waterfall.png          — 无迁移→UDA改善瀑布图
#   figures/dynatopo_radar.png              — 多维度模型对比雷达图
#
# 运行:
#   python notebooks/dynatopo/plot_dynatopo_experiments.py
# ============================================================
import os
import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

sns.set_theme(style='whitegrid', context='paper', font_scale=1.05,
              rc={'axes.edgecolor': '0.15', 'grid.alpha': 0.2,
                  'figure.facecolor': 'white', 'axes.facecolor': '#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Nature 配色
NATURE = {
    'blue': '#4C72B0', 'red': '#C44E52', 'green': '#55A868',
    'orange': '#DD8452', 'purple': '#937860', 'gray': '#8C8C8C',
    'light': '#EAEAF2', 'dark': '#2C2C2C',
}

MODEL_PALETTE = {'static': '#7F8C8D', 'A1B1': '#1F77B4',
                 'A2B1': '#2CA02C', 'A2B2': '#D62728'}
MODEL_LABELS = {'static': 'Static', 'A1B1': 'A1B1', 'A2B1': 'A2B1', 'A2B2': 'A2B2'}
# 包含 A1B2 用于展示
MODEL_PALETTE_5 = {'static': '#7F8C8D', 'A1B1': '#1F77B4',
                   'A1B2': '#FF7F0E', 'A2B1': '#2CA02C', 'A2B2': '#D62728'}

# ---- 加载数据 ----
with open(os.path.join(ROOT, 'experiments', 'logs', 'summary.json'), encoding='utf-8') as f:
    summary_data = json.load(f)

with open(os.path.join(ROOT, 'logs', 'dynatopo', 'eval_cross_condition.json'), encoding='utf-8') as f:
    cross_data = json.load(f)

with open(os.path.join(ROOT, 'logs', 'dynatopo', 'eval_full_FD001.json'), encoding='utf-8') as f:
    fd001_data = json.load(f)


# ============================================================
# 图1: 多 seed 稳定性箱线图 (模拟)
# 用 summary.json 中的 mean/std 生成模拟分布
# ============================================================
def draw_multiseed_stability():
    """绘制多seed指标对比：带误差棒的条形图，突出A1B2的高方差"""
    presets_5 = ['static', 'A1B1', 'A1B2', 'A2B1', 'A2B2']
    metrics = [
        ('test_rmse', 'Test RMSE', '↓'),
        ('test_nasa', 'Test NASA Score', '↓'),
        ('val_rmse', 'Val RMSE', '↓'),
        ('val_nasa', 'Val NASA Score', '↓'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for idx, (key, title, arrow) in enumerate(metrics):
        ax = axes[idx]
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'

        means = [summary_data[p][mean_key] for p in presets_5]
        stds = [summary_data[p][std_key] for p in presets_5]
        colors = [MODEL_PALETTE_5[p] for p in presets_5]

        x = np.arange(len(presets_5))
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='white', lw=1.2,
                      capsize=8, width=0.6, error_kw={'lw': 1.5})

        # 标注 A1B2 的高方差
        ax.annotate(f'[!] 最高方差\nstd={stds[2]:.1f}',
                    xy=(2, means[2]), fontsize=9,
                    ha='center', va='bottom' if idx < 2 else 'top',
                    color=NATURE['red'], fontweight='bold',
                    xytext=(2, means[2] + stds[2] + (max(means)*0.12 if idx < 2 else -max(means)*0.08)),
                    arrowprops=dict(arrowstyle='->', color=NATURE['red'], lw=1.5))

        # 标注最优
        best_idx = np.argmin(means)
        ax.annotate(f'★ 最优\n{means[best_idx]:.2f}',
                    xy=(best_idx, means[best_idx]), fontsize=9,
                    ha='center', va='bottom',
                    color=NATURE['green'], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(presets_5, fontsize=10)
        ax.set_title(f'{title} ({arrow}越低越好)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=10)

        # 在A1B2上加红色虚线框
        rect = plt.Rectangle((1.65, ax.get_ylim()[0]), 0.7, ax.get_ylim()[1] - ax.get_ylim()[0],
                              fill=False, edgecolor=NATURE['red'], lw=2, ls='--', alpha=0.5)
        ax.add_patch(rect)

    fig.suptitle('多 Seed 稳定性对比 (5 seeds, FD001 单工况)', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_multiseed_stability.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_multiseed_stability.png')


# ============================================================
# 图2: 2×2 策略矩阵性能热力图
# ============================================================
def draw_strategy_heatmap():
    """绘制2x2策略矩阵性能热力图：行=生成器, 列=融合策略, 颜色=UDA RMSE"""
    targets = ['FD002', 'FD003', 'FD004']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for t_idx, target in enumerate(targets):
        ax = axes[t_idx]
        # A1B2 已被排除, 位置标记为 N/A
        a1b1 = cross_data[f'A1B1_{target}']['uda_rmse']
        a2b1 = cross_data[f'A2B1_{target}']['uda_rmse']
        a2b2 = cross_data[f'A2B2_{target}']['uda_rmse']

        matrix = np.array([[a1b1, np.nan], [a2b1, a2b2]])
        static_val = cross_data[f'static_{target}']['uda_rmse']

        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap='RdYlGn_r', aspect='auto', vmin=29, vmax=55)

        for i in range(2):
            for j in range(2):
                if i == 0 and j == 1:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=True, facecolor='#EEEEEE', alpha=0.6, zorder=2))
                    ax.text(j, i, 'N/A\n(已排除)', ha='center', va='center',
                            fontsize=10, fontweight='bold', color=NATURE['gray'])
                else:
                    val = matrix[i, j]
                    text_color = 'white' if val > 48 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=14, fontweight='bold', color=text_color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['B1 (特征融合)', 'B2 (拓扑融合)'], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['A1 (相似度)', 'A2 (注意力)'], fontsize=10)
        ax.set_title(f'{target} UDA RMSE\n(static基线={static_val:.1f})', fontsize=13, fontweight='bold')

        # 标出最优 (排除 NaN)
        valid_vals = [a1b1, a2b1, a2b2]
        best_val = min(valid_vals)
        if best_val == a1b1:
            best_r, best_c = 0, 0
        elif best_val == a2b1:
            best_r, best_c = 1, 0
        else:
            best_r, best_c = 1, 1
        rect = plt.Rectangle((best_c - 0.5, best_r - 0.5), 1, 1, fill=False,
                              edgecolor=NATURE['dark'], lw=3, ls='-')
        ax.add_patch(rect)

        plt.colorbar(im, ax=ax, shrink=0.8, label='RMSE')

    fig.suptitle('2x2 策略矩阵 UDA 性能对比 (无监督域自适应)', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_strategy_heatmap.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_strategy_heatmap.png')


# ============================================================
# 图3: 无迁移 → UDA 改善瀑布图
# ============================================================
def draw_waterfall():
    """以 FD002 为例，展示从 static 到各动态图 UDA 的改善路径"""
    target = 'FD002'
    presets = ['static', 'A1B1', 'A2B1', 'A2B2']

    no_transfer = [cross_data[f'{p}_{target}']['no_transfer_rmse'] for p in presets]
    uda = [cross_data[f'{p}_{target}']['uda_rmse'] for p in presets]
    improvement = [nt - u for nt, u in zip(no_transfer, uda)]
    colors = [MODEL_PALETTE[p] for p in presets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 无迁移 vs UDA 对比
    x = np.arange(len(presets))
    width = 0.35
    bars1 = ax1.bar(x - width/2, no_transfer, width, label='无迁移', color='#E0E0E0',
                    edgecolor=NATURE['gray'], lw=1)
    bars2 = ax1.bar(x + width/2, uda, width, label='UDA (无监督)', color=colors,
                    edgecolor='white', lw=1.2)

    for bar, val in zip(bars1, no_transfer):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', fontsize=9, color=NATURE['gray'])
    for bar, val in zip(bars2, uda):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(presets, fontsize=11)
    ax1.set_ylabel('Test RMSE', fontsize=11)
    ax1.set_title(f'{target} 无迁移 vs UDA 对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)

    # 右图: 改善幅度瀑布
    # 按改善幅度排序
    sorted_idx = np.argsort(improvement)[::-1]
    sorted_presets = [presets[i] for i in sorted_idx]
    sorted_imp = [improvement[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    bars = ax2.barh(range(len(sorted_presets)), sorted_imp, color=sorted_colors,
                    edgecolor='white', lw=1.2, height=0.5)
    for bar, val in zip(bars, sorted_imp):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}', ha='left', va='center', fontsize=11, fontweight='bold')

    ax2.set_yticks(range(len(sorted_presets)))
    ax2.set_yticklabels(sorted_presets, fontsize=11)
    ax2.set_xlabel('UDA 改善幅度 (RMSE 降幅)', fontsize=11)
    ax2.set_title(f'{target} UDA 相对无迁移改善', fontsize=13, fontweight='bold')
    ax2.axvline(0, color=NATURE['dark'], lw=1.5, ls='-')

    fig.suptitle(f'无监督域自适应 (UDA) 改善分析 — {target}', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_waterfall.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_waterfall.png')


# ============================================================
# 图4: 多维度雷达图对比
# ============================================================
def draw_radar():
    """多维度雷达图：5轴对比各模型综合表现"""
    presets = ['static', 'A1B1', 'A2B1', 'A2B2']

    # 归一化维度 (越小越好，取倒数后越大越好)
    # 维度: test_rmse, test_nasa, val_rmse, val_nasa, stability(1/std)
    raw = {}
    for p in presets:
        s = summary_data[p]
        # 用 test_rmse_std 作为稳定性代理
        stability = 1.0 / (s['test_rmse_std'] + 0.01)
        raw[p] = {
            'Test RMSE ↓': s['test_rmse_mean'],
            'Test NASA ↓': s['test_nasa_mean'],
            'Val RMSE ↓': s['val_rmse_mean'],
            'Val NASA ↓': s['val_nasa_mean'],
            '稳定性 ↑': stability,
        }

    # 归一化到 [0,1] 区间 (越小越好→取反)
    dims = ['Test RMSE ↓', 'Test NASA ↓', 'Val RMSE ↓', 'Val NASA ↓', '稳定性 ↑']
    norm = {}
    for dim in dims:
        vals = [raw[p][dim] for p in presets]
        min_v, max_v = min(vals), max(vals)
        if dim == '稳定性 ↑':
            for p in presets:
                norm.setdefault(p, {})[dim] = (raw[p][dim] - min_v) / (max_v - min_v + 1e-8)
        else:
            for p in presets:
                norm.setdefault(p, {})[dim] = (max_v - raw[p][dim]) / (max_v - min_v + 1e-8)

    # 绘制雷达图
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for p in presets:
        values = [norm[p][d] for d in dims]
        values += values[:1]
        ax.fill(angles, values, alpha=0.1, color=MODEL_PALETTE[p])
        ax.plot(angles, values, 'o-', linewidth=2, color=MODEL_PALETTE[p], label=p, markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color=NATURE['gray'])
    ax.set_title('多维度模型综合对比 (归一化)', fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_radar.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_radar.png')


# ============================================================
if __name__ == '__main__':
    draw_multiseed_stability()
    draw_strategy_heatmap()
    draw_waterfall()
    draw_radar()
    print('Done! All experiment figures generated.')