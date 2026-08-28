# ============================================================
# plot_dynatopo_architecture.py
# 动态拓扑 STGNN 架构总览图 & NASA 非对称惩罚函数图
#
# 产出:
#   figures/dynatopo_architecture.png  — 动态 STGNN 完整架构图（概念图）
#   figures/dynatopo_nasa_penalty.png  — NASA 非对称惩罚函数曲线
#
# 运行:
#   python notebooks/dynatopo/plot_dynatopo_architecture.py
# ============================================================
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Nature 配色
NATURE = {
    'blue': '#4C72B0', 'red': '#C44E52', 'green': '#55A868',
    'orange': '#DD8452', 'purple': '#937860', 'gray': '#8C8C8C',
    'light': '#EAEAF2', 'dark': '#2C2C2C', 'yellow': '#E8C854'
}


# ============================================================
# 图1: 动态 STGNN (DynaTopo) 完整架构图
# ============================================================
def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')

    def add_box(x, y, w, h, text, color=NATURE['blue'], fs=10, fc='white', ec=None, lw=2):
        if ec is None:
            ec = color
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
                fontweight='bold', color=color, zorder=3)
        return box

    def add_arrow(x1, y1, x2, y2, color=NATURE['gray'], lw=1.5, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    connectionstyle='arc3,rad=0'))

    # ---- 标题 ----
    ax.text(9, 9.5, '动态图时空图神经网络 (DynaTopo-STGNN) 架构总览',
            ha='center', va='center', fontsize=16, fontweight='bold', color=NATURE['dark'])

    # ===== 左侧: 输入 =====
    add_box(0.5, 7.2, 2.5, 1.5, '多传感器时序输入\nX ∈ R^{B×N×W}\n(N=14传感器, W=30窗口)',
            NATURE['dark'], fs=9, lw=1.5)

    # ===== MSTCN =====
    add_box(0.5, 4.5, 2.5, 2.0, 'MSTCN 多尺度时序编码\n━━━━━━━━━━━━━━\nConv1d(k=3,ch=32) 短期\nConv1d(k=5,ch=64) 中期\nConv1d(k=7,ch=128) 长期\n→ H ∈ R^{B×N×128}',
            NATURE['blue'], fs=8, lw=1.5)

    # ===== 中间: 双图分支 =====
    # 静态图分支
    add_box(4.0, 6.5, 3.0, 2.0,
            '静态拓扑图 (Spearman)\n━━━━━━━━━━━━━━\n- 训练前一次性计算\n- |ρ| > 0.6 阈值建边\n- 固定62条边\n- 与工况无关',
            NATURE['gray'], fs=8, fc='#F5F5F5', lw=1.5)

    # 动态图分支
    add_box(4.0, 3.5, 3.0, 2.5,
            '动态拓扑图 (工况驱动)\n━━━━━━━━━━━━━━\nA1: 余弦相似度 (无参)\nA2: 多头注意力 (可训练)\n- 每个样本实时生成\n- 随工况参数(op1~op3)自适应\n- 动态K近邻图',
            NATURE['orange'], fs=8, fc='#FFF8F0', lw=1.5)

    # 工况参数输入
    add_box(4.0, 1.5, 3.0, 1.2,
            '工况参数 op1, op2, op3\n(海拔/马赫数/油门)',
            NATURE['purple'], fs=8, fc='#F8F0FF', lw=1.5)

    # ===== 融合策略 =====
    add_box(8.0, 6.5, 2.2, 2.0,
            'B1: 特征层融合\n━━━━━━━━\n静态图GAT → h_static\n动态图GAT → h_dyn\n加权融合:\nh = α·h_static + (1-α)·h_dyn',
            NATURE['green'], fs=8, fc='#F0FFF0', lw=1.5)

    add_box(8.0, 3.8, 2.2, 2.2,
            'B2: 拓扑层融合\n━━━━━━━━\n合并静态边+动态边\n→ 统一GAT\n→ 单次消息传递',
            NATURE['red'], fs=8, fc='#FFF0F0', lw=1.5)

    # ===== 右侧: GAT + 预测 =====
    add_box(11.0, 6.0, 2.5, 2.5,
            'GAT 图注意力网络\n━━━━━━━━━━━━━━\n• 多头注意力(4头)\n• 节点间消息传递\n• 传感器空间关系建模\n→ h_fused ∈ R^{B×N×D}',
            NATURE['blue'], fs=8, lw=1.5)

    add_box(11.0, 2.8, 2.5, 2.0,
            '全局池化 + 预测头\n━━━━━━━━━━━━━━\n• 全局平均池化\n• MLP(128→64→1)\n→ RUL 预测值',
            NATURE['dark'], fs=8, fc='#FCFCFC', lw=1.5)

    # ===== 最右侧: 损失 =====
    add_box(14.2, 3.5, 3.0, 3.5,
            '训练目标\n━━━━━━━━━━━━━━\n• 源域: MSE Loss\n• 目标域(UDA):\n  LMMD 子域对齐损失\n  L = L_mse + λ·L_lmmd\n━━━━━━━━━━━━━━\n• 4种迁移模式:\n  无迁移 / 监督式 / \n  无监督UDA / 微调',
            NATURE['red'], fs=8, fc='#FFF5F5', lw=1.5)

    # ===== 箭头连接 =====
    # 输入 → MSTCN
    add_arrow(1.75, 7.2, 1.75, 6.5, NATURE['gray'], 1.5)
    # MSTCN → 静态图
    add_arrow(2.5, 5.8, 4.0, 7.5, NATURE['gray'], 1.2)
    # MSTCN → 动态图
    add_arrow(2.5, 5.0, 4.0, 5.0, NATURE['orange'], 1.2)
    # 工况 → 动态图
    add_arrow(5.5, 2.7, 5.5, 3.5, NATURE['purple'], 1.0)
    # 静态图 → B1
    add_arrow(7.0, 7.5, 8.0, 7.5, NATURE['gray'], 1.2)
    # 动态图 → B1
    add_arrow(7.0, 5.0, 8.0, 7.0, NATURE['orange'], 1.2)
    # 静态图 → B2
    add_arrow(7.0, 7.0, 8.0, 5.5, NATURE['gray'], 1.0)
    # 动态图 → B2
    add_arrow(7.0, 4.5, 8.0, 4.8, NATURE['orange'], 1.0)
    # B1 → GAT
    add_arrow(10.2, 7.5, 11.0, 7.5, NATURE['green'], 1.2)
    # B2 → GAT
    add_arrow(10.2, 5.0, 11.0, 6.5, NATURE['red'], 1.2)
    # GAT → 预测头
    add_arrow(12.25, 6.0, 12.25, 4.8, NATURE['blue'], 1.2)
    # 预测头 → 损失
    add_arrow(13.5, 3.8, 14.2, 4.5, NATURE['dark'], 1.0)

    # ===== 图例 =====
    legend_items = [
        mlines.Line2D([], [], color=NATURE['gray'], lw=2, label='静态拓扑分支'),
        mlines.Line2D([], [], color=NATURE['orange'], lw=2, label='动态拓扑分支'),
        mlines.Line2D([], [], color=NATURE['purple'], lw=2, label='工况参数输入'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=9,
              frameon=True, fancybox=True)

    fig.tight_layout(pad=1)
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_architecture.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_architecture.png')


# ============================================================
# 图2: NASA 非对称惩罚函数曲线
# ============================================================
def draw_nasa_penalty():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- 左图: 惩罚函数 ----
    d = np.linspace(-50, 50, 500)
    penalty = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)

    ax1.fill_between(d[d < 0], 0, penalty[d < 0], alpha=0.15, color=NATURE['green'], label='过早预测 (惩罚较轻)')
    ax1.fill_between(d[d > 0], 0, penalty[d > 0], alpha=0.15, color=NATURE['red'], label='过晚预测 (惩罚较重)')
    ax1.plot(d, penalty, color=NATURE['dark'], lw=2.5)
    ax1.axvline(0, color=NATURE['gray'], ls='--', lw=1, alpha=0.6)
    ax1.axhline(0, color=NATURE['gray'], ls='-', lw=0.5, alpha=0.3)

    # 标注关键点
    for d_val in [-20, -10, 10, 20]:
        p = np.exp(abs(d_val) / (13 if d_val < 0 else 10)) - 1
        ax1.annotate(f'd={d_val}\npen={p:.1f}',
                     xy=(d_val, p), fontsize=8,
                     ha='center', va='bottom' if d_val < 0 else 'top',
                     color=NATURE['green'] if d_val < 0 else NATURE['red'],
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax1.set_xlabel('预测误差 d = 预测RUL − 真实RUL', fontsize=11)
    ax1.set_ylabel('NASA 惩罚值', fontsize=11)
    ax1.set_title('C-MAPSS 非对称评分函数', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-2, 60)

    # ---- 右图: 直观对比 ----
    scenarios = ['过早预测\n(保守)', '精确预测', '过晚预测\n(危险)']
    pred_vals = [80, 100, 120]
    true_val = 100
    errors = [p - true_val for p in pred_vals]
    nasa_scores = [np.exp(abs(e)/(13 if e < 0 else 10)) - 1 for e in errors]
    colors = [NATURE['green'], NATURE['blue'], NATURE['red']]

    x_pos = np.arange(3)
    bars = ax2.bar(x_pos, nasa_scores, color=colors, edgecolor='white', lw=1.5, width=0.5)

    for i, (bar, s, e, p) in enumerate(zip(bars, scenarios, errors, pred_vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'Score={nasa_scores[i]:.1f}\n预测={p}, 误差={e:+d}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.set_ylabel('NASA Score', fontsize=11)
    ax2.set_title(f'相同 |误差|=20 时的惩罚差异\n(真实RUL={true_val})', fontsize=13, fontweight='bold')
    ax2.axhline(0, color=NATURE['gray'], ls='-', lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dynatopo_nasa_penalty.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] dynatopo_nasa_penalty.png')


# ============================================================
if __name__ == '__main__':
    draw_architecture()
    draw_nasa_penalty()
    print('Done! All architecture figures generated.')