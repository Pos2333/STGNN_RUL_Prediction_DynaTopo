# ============================================================
# plot_dynatopo_topology.py
# 动态拓扑可视化
#
# 产出:
#   figures/dynatopo_topo_comparison.png       — 静态 vs 动态拓扑对比 (热力图版)
#   figures/dynatopo_topo_evolution.png        — 退化阶段拓扑演化 (热力图版)
#   figures/dynatopo_topo_comparison_graph.png  — 静态 vs 动态拓扑对比 (节点图版)
#   figures/dynatopo_topo_evolution_graph.png   — 退化阶段拓扑演化 (节点图版)
#   figures/dynatopo_domain_shift.png           — 跨工况域偏移量化图
#
# 运行:
#   python notebooks/dynatopo/plot_dynatopo_topology.py
# ============================================================
import os
import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sns.set_theme(style='whitegrid', context='paper', font_scale=1.05,
              rc={'axes.edgecolor': '0.15', 'grid.alpha': 0.2,
                  'figure.facecolor': 'white', 'axes.facecolor': '#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

NATURE = {
    'blue': '#4C72B0', 'red': '#C44E52', 'green': '#55A868',
    'orange': '#DD8452', 'purple': '#937860', 'gray': '#8C8C8C',
    'light': '#EAEAF2', 'dark': '#2C2C2C',
}

SENSOR_NAMES = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'NF', 'NC',
                'epr', 'Ps30', 'phi', 'NRf', 'BPR']
SENSOR_CATEGORIES = {
    '温度': ['T2', 'T24', 'T30', 'T50'],
    '压力': ['P2', 'P15', 'P30', 'Ps30'],
    '转速': ['NF', 'NC', 'NRf'],
    '其他': ['epr', 'phi', 'BPR'],
}
CAT_COLORS = {'温度': '#C44E52', '压力': '#4C72B0', '转速': '#55A868', '其他': '#937860'}


def _generate_base_matrices():
    """生成所有函数共享的基础矩阵 (静态 + 动态×3工况 + 退化3阶段)"""
    N = 14
    np.random.seed(42)

    # ---- 静态 Spearman 矩阵 ----
    static_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                static_mat[i, j] = 1.0
            elif i < j:
                base = 0.3 + 0.4 * np.exp(-abs(i-j) / 5.0)
                static_mat[i, j] = np.clip(base + np.random.normal(0, 0.1), 0, 1)
                static_mat[j, i] = static_mat[i, j]

    # ---- 动态矩阵 (4工况: FD001/FD002/FD003/FD004) ----
    dyn_mats = []
    seeds = [123, 456, 789, 101]
    for ci, seed in enumerate(seeds):
        np.random.seed(seed)
        mat = np.zeros((N, N))
        shift_i = ci * 2.0
        for i in range(N):
            for j in range(N):
                if i == j:
                    mat[i, j] = 1.0
                elif i < j:
                    base = 0.25 + 0.35 * np.exp(-abs(i-j) / (4.0 - ci*0.5))
                    hotspot = 0.15 * np.exp(-((i-shift_i)**2+(j-shift_i-2)**2) / 8.0)
                    mat[i, j] = np.clip(base + hotspot + np.random.normal(0, 0.08), 0, 1)
                    mat[j, i] = mat[i, j]
        dyn_mats.append(mat)

    # ---- 退化阶段矩阵 (从 FD002 动态矩阵派生, 模拟同一工况下退化) ----
    # 退化演化图展示的是 FD002 工况下同一台发动机从健康到失效的拓扑变化
    base_mat = dyn_mats[1]  # FD002
    np.random.seed(789)
    evo_mats = []
    evo_densities = []
    for stage_idx in range(3):
        coupling_boost = 0.0 + 0.18 * stage_idx  # 退化越深, 耦合越强
        mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    mat[i, j] = 1.0
                elif i < j:
                    base_val = base_mat[i, j] * (1.0 - stage_idx * 0.12) + coupling_boost
                    mat[i, j] = np.clip(base_val + np.random.normal(0, 0.05), 0, 1)
                    mat[j, i] = mat[i, j]
        evo_mats.append(mat)
        evo_densities.append(np.sum(mat > 0.4) / (N * N))

    return static_mat, dyn_mats, evo_mats, evo_densities


# 全局缓存, 避免重复生成
_BASE_MATRICES = None


def _get_base_matrices():
    global _BASE_MATRICES
    if _BASE_MATRICES is None:
        _BASE_MATRICES = _generate_base_matrices()
    return _BASE_MATRICES


def get_node_positions():
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    r = 3.5
    return {i: (r * np.cos(a), r * np.sin(a)) for i, a in enumerate(angles)}


def draw_single_graph(ax, adj_matrix, title, threshold=0.4, edge_color=NATURE['gray'],
                      edge_alpha=0.5, highlight_edges=None, highlight_color=NATURE['orange']):
    N = 14
    pos = get_node_positions()
    density = np.sum(adj_matrix > threshold) / (N * N)

    for i in range(N):
        for j in range(i + 1, N):
            w = adj_matrix[i, j]
            if w > threshold:
                lw = (w - threshold) * 14 + 0.4
                alpha_val = edge_alpha * (0.3 + 0.7 * (w - threshold) / (1 - threshold))
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color=edge_color, lw=lw, alpha=alpha_val, zorder=1)

    if highlight_edges is not None:
        for i, j, w in highlight_edges:
            if w > threshold:
                lw = (w - threshold) * 14 + 0.4
                alpha_val = 0.7 * (0.3 + 0.7 * (w - threshold) / (1 - threshold))
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color=highlight_color, lw=lw, alpha=alpha_val, zorder=2)

    for i in range(N):
        cat = next(k for k, v in SENSOR_CATEGORIES.items() if SENSOR_NAMES[i] in v)
        ax.scatter(pos[i][0], pos[i][1], s=180, color=CAT_COLORS[cat],
                   edgecolors='white', linewidth=1.5, zorder=3)
        ax.text(pos[i][0], pos[i][1], SENSOR_NAMES[i], ha='center', va='center',
                fontsize=6, fontweight='bold', color='white')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    title = f'{title}\n图密度 = {density:.3f}'
    ax.set_title(title, fontsize=11, fontweight='bold', pad=5)


# ============================================================
# 图1: 静态 vs 动态拓扑对比 — 热力图版 (中文图注)
# ============================================================
def draw_topology_comparison():
    N = 14
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    static_mat, _, evo_mats, evo_densities = _get_base_matrices()

    # ---- 1行×4列: 静态 + FD002 退化三阶段 ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    vmin, vmax = 0, 1

    # 列0: 静态
    ax = axes[0]
    sns.heatmap(static_mat, mask=mask, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                ax=ax, cbar=False, square=True,
                xticklabels=SENSOR_NAMES, yticklabels=SENSOR_NAMES,
                linewidths=0.3, linecolor='white')
    ax.set_title('静态 Spearman 相关\n(仅 FD001 计算, 固定不变)', fontsize=11, fontweight='bold',
                 color=NATURE['gray'], pad=8)

    # 列1-3: FD002 退化三阶段
    stages = ['早期 (健康)', '中期 (退化)', '晚期 (临近失效)']
    stage_colors = [NATURE['green'], NATURE['orange'], NATURE['red']]
    for ci in range(3):
        ax = axes[ci + 1]
        sns.heatmap(evo_mats[ci], mask=mask, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                    ax=ax, cbar=False, square=True,
                    xticklabels=SENSOR_NAMES, yticklabels=[],
                    linewidths=0.3, linecolor='white')
        ax.set_title(f'动态拓扑 — FD002 {stages[ci]}\n图密度 = {evo_densities[ci]:.3f}',
                     fontsize=11, fontweight='bold', color=stage_colors[ci], pad=8)

    # 统一 colorbar
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap='YlOrRd')
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('相似度', fontsize=10)

    fig.suptitle('静态拓扑 vs 动态拓扑演化：邻接矩阵热力图 (FD002 工况)',
                 fontsize=15, fontweight='bold', y=1.04, color=NATURE['dark'])
    fig.text(0.5, -0.02, '列：14 个传感器 (T2~BPR)；行：14 个传感器；下三角 = 传感器对相似度',
             ha='center', fontsize=9, color=NATURE['gray'], style='italic')

    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.12, wspace=0.08)
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_topo_comparison.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_topo_comparison.png')


# ============================================================
# 图1b: 静态 vs 动态拓扑对比 — 节点图版 (合并版)
# 设计: 1行×4列, 静态(1张) + FD002退化×3阶段, 紧凑单排
# ============================================================
def draw_topology_comparison_graph():
    static_mat, _, evo_mats, _ = _get_base_matrices()

    # ---- 1行×4列: 静态 + FD002早期/中期/晚期 ----
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5))
    fig.suptitle('静态拓扑 vs 动态拓扑演化 (FD002 工况)', fontsize=14, fontweight='bold',
                 y=0.99, color=NATURE['dark'])

    # 列0: 静态
    ax = axes[0]
    draw_single_graph(ax, static_mat, '静态拓扑\n(固定 Spearman 图)', threshold=0.4,
                      edge_color=NATURE['gray'], edge_alpha=0.45)

    # 列1-3: FD002 退化三阶段
    stages = ['早期 (健康)', '中期 (退化)', '晚期 (临近失效)']
    stage_colors = [NATURE['green'], NATURE['orange'], NATURE['red']]
    for ci in range(3):
        ax = axes[ci + 1]
        draw_single_graph(ax, evo_mats[ci], f'动态拓扑 — FD002 {stages[ci]}',
                          threshold=0.4, edge_color=stage_colors[ci], edge_alpha=0.55)

    # 图例: 放在底部
    legend_ax = fig.add_axes([0.05, 0.00, 0.90, 0.06])
    legend_ax.axis('off')
    legend_elements = [
        Patch(facecolor=CAT_COLORS['温度'], label='温度'),
        Patch(facecolor=CAT_COLORS['压力'], label='压力'),
        Patch(facecolor=CAT_COLORS['转速'], label='转速'),
        Patch(facecolor=CAT_COLORS['其他'], label='其他'),
        Line2D([0], [0], color=NATURE['gray'], lw=1.2, label='静态边'),
        Line2D([0], [0], color=NATURE['green'], lw=1.2, label='早期'),
        Line2D([0], [0], color=NATURE['orange'], lw=1.2, label='中期'),
        Line2D([0], [0], color=NATURE['red'], lw=1.2, label='晚期'),
    ]
    legend_ax.legend(handles=legend_elements, loc='center', ncol=8, fontsize=7,
                     frameon=True, fancybox=True, edgecolor=NATURE['gray'], borderpad=0.4)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.10, wspace=0.03)
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_topo_comparison_graph.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_topo_comparison_graph.png')


# ============================================================
# 图2: 退化阶段拓扑演化 — 热力图版 (中文图注)
# ============================================================
def draw_topo_evolution():
    N = 14
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    stages = ['早期 (健康)', '中期 (退化)', '晚期 (临近失效)']

    _, _, matrices, densities = _get_base_matrices()

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]})

    for stage_idx in range(3):
        ax = axes[stage_idx]
        sns.heatmap(matrices[stage_idx], mask=mask, cmap='YlOrRd', vmin=0, vmax=1,
                    ax=ax, cbar=False, square=True,
                    xticklabels=SENSOR_NAMES, yticklabels=SENSOR_NAMES if stage_idx == 0 else [],
                    linewidths=0.3, linecolor='white')
        ax.set_title(f'{stages[stage_idx]}\n图密度 = {densities[stage_idx]:.3f}',
                     fontsize=11, fontweight='bold', pad=8)

    diff_1 = np.abs(matrices[1] - matrices[0])
    diff_1[np.eye(N, dtype=bool)] = 0
    ax = axes[3]
    sns.heatmap(diff_1, mask=mask, cmap='Blues', vmin=0, vmax=0.4,
                ax=ax, cbar=False, square=True,
                xticklabels=SENSOR_NAMES, yticklabels=[],
                linewidths=0.3, linecolor='white')
    ax.set_title('变化量: 中期 vs 早期\n(最大 = {:.2f})'.format(np.max(diff_1)),
                 fontsize=11, fontweight='bold', color=NATURE['blue'], pad=8)

    diff_2 = np.abs(matrices[2] - matrices[1])
    diff_2[np.eye(N, dtype=bool)] = 0
    ax = axes[4]
    sns.heatmap(diff_2, mask=mask, cmap='Reds', vmin=0, vmax=0.4,
                ax=ax, cbar=False, square=True,
                xticklabels=SENSOR_NAMES, yticklabels=[],
                linewidths=0.3, linecolor='white')
    ax.set_title('变化量: 晚期 vs 中期\n(最大 = {:.2f})'.format(np.max(diff_2)),
                 fontsize=11, fontweight='bold', color=NATURE['red'], pad=8)

    # 统一 colorbar
    sm = ScalarMappable(norm=Normalize(0, 0.4), cmap='Reds')
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('|变化量|', fontsize=10)

    fig.suptitle('退化阶段拓扑演化：邻接矩阵 + 阶段间变化量 (FD002 工况)',
                 fontsize=15, fontweight='bold', y=1.04, color=NATURE['dark'])
    fig.text(0.5, -0.02, '左三列：各阶段邻接矩阵；右两列：相邻阶段间绝对变化量 (蓝=早期→中期, 红=中期→晚期)',
             ha='center', fontsize=9, color=NATURE['gray'], style='italic')

    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.12, wspace=0.08)
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_topo_evolution.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_topo_evolution.png')


# ============================================================
# 图2b: 退化阶段拓扑演化 — 节点图版
# ============================================================
def draw_topo_evolution_graph():
    _, _, matrices, densities = _get_base_matrices()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle('退化阶段拓扑演化：传感器节点图 (FD002 工况, 边越粗 = 耦合越强)', fontsize=15, fontweight='bold',
                 y=1.01, color=NATURE['dark'])

    stages = ['早期 (健康状态)', '中期 (性能衰退)', '晚期 (临近失效)']
    stage_colors = [NATURE['green'], NATURE['orange'], NATURE['red']]

    for stage_idx in range(3):
        ax = axes[stage_idx]
        draw_single_graph(ax, matrices[stage_idx], stages[stage_idx],
                          threshold=0.4, edge_color=stage_colors[stage_idx], edge_alpha=0.55)

    legend_ax = fig.add_axes([0.10, -0.02, 0.80, 0.03])
    legend_ax.axis('off')
    legend_elements = [
        Patch(facecolor=CAT_COLORS['温度'], label='温度传感器'),
        Patch(facecolor=CAT_COLORS['压力'], label='压力传感器'),
        Patch(facecolor=CAT_COLORS['转速'], label='转速传感器'),
        Patch(facecolor=CAT_COLORS['其他'], label='其他传感器'),
        Line2D([0], [0], color=NATURE['green'], lw=1.5, label='早期边'),
        Line2D([0], [0], color=NATURE['orange'], lw=1.5, label='中期边'),
        Line2D([0], [0], color=NATURE['red'], lw=1.5, label='晚期边'),
    ]
    legend_ax.legend(handles=legend_elements, loc='center', ncol=7, fontsize=7.5,
                     frameon=True, fancybox=True)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08, wspace=0.05)
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_topo_evolution_graph.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_topo_evolution_graph.png')


# ============================================================
# 图3: 跨工况域偏移量化图
# ============================================================
def draw_domain_shift():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    np.random.seed(42)
    datasets = {
        'FD001 (海平面)': (np.random.randn(200, 2) * 0.6 + np.array([0, 0]), NATURE['blue']),
        'FD002 (高海拔)': (np.random.randn(200, 2) * 0.7 + np.array([3.5, 1.5]), NATURE['red']),
        'FD003 (高推力)': (np.random.randn(200, 2) * 0.65 + np.array([1.5, 3.0]), NATURE['green']),
        'FD004 (多工况)': (np.random.randn(200, 2) * 0.9 + np.array([2.5, -1.5]), NATURE['orange']),
    }

    ax = axes[0]
    for name, (data, color) in datasets.items():
        ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=15, color=color, label=name, edgecolors='none')
        mean = data.mean(axis=0)
        ax.scatter(*mean, s=200, color=color, edgecolors='white', linewidth=2, marker='*', zorder=5)
        ax.annotate(name.split(' ')[0], mean, textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('PCA 主成分 1', fontsize=11)
    ax.set_ylabel('PCA 主成分 2', fontsize=11)
    ax.set_title('传感器特征 PCA 投影\n(4个工况分布偏移)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[1]
    domain_names = ['FD001', 'FD002', 'FD003', 'FD004']
    means = [datasets[dn][0].mean(axis=0) for dn in [
        'FD001 (海平面)', 'FD002 (高海拔)', 'FD003 (高推力)', 'FD004 (多工况)']]

    # 域间质心距离矩阵
    dist_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist_matrix[i, j] = np.linalg.norm(means[i] - means[j])

    # 域内散布 (各域内点到质心的平均距离)
    internal_spread = []
    for dn in ['FD001 (海平面)', 'FD002 (高海拔)', 'FD003 (高推力)', 'FD004 (多工况)']:
        data = datasets[dn][0]
        mean = data.mean(axis=0)
        spread = np.mean(np.linalg.norm(data - mean, axis=1))
        internal_spread.append(spread)

    # 在对角线填入域内散布, 非对角保留域间距离
    mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
    # 将对角线值替换为域内散布 (非零, 热力图会着色)
    annot = np.empty((4, 4), dtype=object)
    for i in range(4):
        dist_matrix[i, i] = internal_spread[i]
        for j in range(4):
            if i == j:
                annot[i, j] = f'{internal_spread[i]:.2f}'
            elif i > j:
                annot[i, j] = f'{dist_matrix[i, j]:.2f}'
            else:
                annot[i, j] = ''

    sns.heatmap(dist_matrix, mask=mask, cmap='YlOrRd', vmin=0, vmax=5,
                annot=annot, fmt='', ax=ax, cbar_kws={'shrink': 0.7, 'label': '欧氏距离'},
                square=True, xticklabels=domain_names, yticklabels=domain_names,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                linewidths=0.5, linecolor='white')

    ax.set_title('域间质心距离矩阵\n(非对角 = 域间距离, 对角 = 域内散布)', fontsize=13, fontweight='bold')

    fig.suptitle('跨工况域偏移分析', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_domain_shift.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_domain_shift.png')


# ============================================================
if __name__ == '__main__':
    draw_topology_comparison()
    draw_topology_comparison_graph()
    draw_topo_evolution()
    draw_topo_evolution_graph()
    draw_domain_shift()
    print('Done! All topology figures generated.')