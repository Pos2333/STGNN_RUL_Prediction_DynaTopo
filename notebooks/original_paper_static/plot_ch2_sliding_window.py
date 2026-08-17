# ============================================================
# plot_ch2_sliding_window.py
# 第2章 滑动窗口样本构建示意图 —— 演示变长轨迹 → 固定窗口样本的机理
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# === Seaborn 学术风格 ===
sns.set_theme(style='whitegrid', context='paper', font_scale=1.1,
              rc={'axes.edgecolor':'0.15','grid.alpha':0.2,
                  'figure.facecolor':'white','axes.facecolor':'#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
from configs.config import WINDOW_SIZE, RUL_CLIP_MAX, NUM_FEATURES, RANDOM_SEED
np.random.seed(RANDOM_SEED)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# === Nature 配色 ===
C_WINDOW_1  = '#4C72B0'   # Blue   — 窗口1 (早期)
C_WINDOW_2  = '#55A868'   # Green  — 窗口2 (中期)
C_WINDOW_3  = '#DD8452'   # Orange — 窗口3 (退化)
C_WINDOW_4  = '#C44E52'   # Red    — 窗口4 (临近失效)
C_ENGINE_BG = '#EAEAF2'   # Light  — 发动机背景
C_ARROW     = '#2C2C2C'   # Dark   — 箭头和文字
C_LABEL_BG  = '#FFF8E1'   # Yellow — 标签背景
WINDOW_COLORS = [C_WINDOW_1, C_WINDOW_2, C_WINDOW_3, C_WINDOW_4]

# === 选取代表性发动机 ===
SELECTED_ENGINES = [1, 2, 91]  # 中等(192) / 长(287) / 短(135)
ENGINE_INFO = {
    1:  {'life': 192, 'label': 'Engine #1\n(192 cycles)'},
    2:  {'life': 287, 'label': 'Engine #2\n(287 cycles)'},
    91: {'life': 135, 'label': 'Engine #91\n(135 cycles)'},
}

# === 演示窗口位置 (Engine #1, 192 cycles) ===
# 选取 4 个代表性窗口起止位置，覆盖完整生命周期
DEMO_WINDOWS = [
    {'start': 1,   'label': '早期健康', 'desc': 'cycles\n1–30'},
    {'start': 60,  'label': '中期退化', 'desc': 'cycles\n60–89'},
    {'start': 120, 'label': '明显退化', 'desc': 'cycles\n120–149'},
    {'start': 162, 'label': '临近失效', 'desc': 'cycles\n162–191'},
]

print("=" * 60)
print("  第2章: 滑动窗口样本构建示意图")
print("=" * 60)

# === 加载 FD001 训练集 ===
RAW_DIR = os.path.join(PROJ_ROOT, 'data', 'raw')
df = pd.read_csv(os.path.join(RAW_DIR, 'train_FD001.txt'), sep=r'\s+', header=None)

# 计算每台发动机的样本数
engine_sample_counts = {}
for eng_id in [1, 2, 91]:
    life = int(df[df[0] == eng_id][1].max())
    num_samples = max(0, life - WINDOW_SIZE + 1)
    engine_sample_counts[eng_id] = {'life': life, 'samples': num_samples}
    print(f"  Engine #{eng_id}: 寿命={life}, 窗口样本数={num_samples}")

# ====================================================================
# 主图: 滑动窗口机制详解 (Engine #1, 192 cycles)
# 布局: 上部=时间轴+窗口示意, 下部=样本统计对比
# ====================================================================
fig = plt.figure(figsize=(19, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.35,
                       top=0.93, bottom=0.08, left=0.07, right=0.97)

# ========================
# 上子图: 滑动窗口示意
# ========================
ax_top = fig.add_subplot(gs[0])
ax_top.set_xlim(-5, 200)
ax_top.set_ylim(-0.5, 6.5)
ax_top.set_yticks([])
ax_top.set_xlabel('运行周期 (Cycle)', fontsize=14, fontweight='bold', labelpad=8)
ax_top.set_title('滑动窗口样本构建示意 ($W = 30$, 以 Engine #1 为例)',
                 fontsize=16, fontweight='bold', pad=15)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['left'].set_visible(False)

# --- 发动机完整轨迹背景条 ---
bar = FancyBboxPatch((-0.5, 4.5), 193, 1.2, boxstyle="round,pad=0.08",
                      facecolor=C_ENGINE_BG, edgecolor='#AAAAAA', linewidth=1.2)
ax_top.add_patch(bar)
ax_top.text(192/2, 5.1, f'Engine #1 完整运行轨迹 (总寿命 192 cycles)',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#444444')
ax_top.text(192/2, 4.45, '每台发动机轨迹长度不同 → 需统一处理为固定长度输入',
            ha='center', va='center', fontsize=10.5, color='#666666', style='italic')

# --- 绘制 4 个示例窗口 ---
for idx, win in enumerate(DEMO_WINDOWS):
    s, e = win['start'], win['start'] + WINDOW_SIZE - 1
    y_pos = 3.8 - idx * 0.9
    color = WINDOW_COLORS[idx]

    # 窗口矩形
    rect = FancyBboxPatch((s - 0.3, y_pos - 0.25), WINDOW_SIZE + 0.6, 0.5,
                           boxstyle="round,pad=0.04", facecolor=color,
                           edgecolor='white', linewidth=1.5, alpha=0.75)
    ax_top.add_patch(rect)

    # 窗口起止标注
    ax_top.annotate('', xy=(s, y_pos - 0.45), xytext=(s, y_pos + 0.45),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax_top.annotate('', xy=(e, y_pos - 0.45), xytext=(e, y_pos + 0.45),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax_top.text(s, y_pos - 0.65, f'$t$={s}', ha='center', va='top',
                fontsize=9, color=color, fontweight='bold')
    ax_top.text(e, y_pos - 0.65, f'$t$+{WINDOW_SIZE-1}={e}', ha='center', va='top',
                fontsize=9, color=color, fontweight='bold')

    # 窗口标签: 名称 + RUL
    rul = max(0, min(192 - e, RUL_CLIP_MAX))
    win_label = f'窗口{idx+1}: {win["label"]}'
    ax_top.text(s - 2, y_pos, win_label, ha='right', va='center',
                fontsize=11, color=color, fontweight='bold')
    ax_top.text(e + 2, y_pos,
                f'→ RUL={rul}',
                ha='left', va='center', fontsize=10.5, color=C_WINDOW_4,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                          edgecolor='#FFB74D', alpha=0.85))

# --- 右侧: 样本矩阵结构示意 ---
matrix_x, matrix_y = 202, 2.5
ax_top.text(matrix_x, matrix_y + 1.8, '每个窗口\n输出为:', ha='center', va='bottom',
            fontsize=10, color='#555555')

# 绘制小矩阵示意图
mat_w, mat_h = 8, 5
rect = FancyBboxPatch((matrix_x - mat_w/2, matrix_y - mat_h/2), mat_w, mat_h,
                       boxstyle="round,pad=0.15", facecolor='white',
                       edgecolor='#333333', linewidth=2)
ax_top.add_patch(rect)
ax_top.text(matrix_x, matrix_y + mat_h/2 + 0.25,
            '$X \\in \\mathbb{R}^{30 \\times 17}$',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#222222')
ax_top.text(matrix_x, matrix_y,
            '30 个时间步\n× 17 维特征\n(3 op + 14 sensor)',
            ha='center', va='center', fontsize=9.5, color='#555555', linespacing=1.6)

# --- 箭头: 窗口 → 样本 ---
ax_top.annotate('', xy=(matrix_x - mat_w/2 - 1, matrix_y),
                xytext=(193, 1.2),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8,
                                connectionstyle='arc3,rad=0.3'))

# --- 图例 ---
legend_patches = []
for idx, win in enumerate(DEMO_WINDOWS):
    legend_patches.append(mpatches.Patch(color=WINDOW_COLORS[idx], alpha=0.75,
                                          label=f'窗口{idx+1}: {win["desc"]}'))
legend = ax_top.legend(handles=legend_patches, loc='lower left',
                        fontsize=9.5, ncol=4, framealpha=0.9,
                        bbox_to_anchor=(0.0, -0.35))
legend.get_frame().set_edgecolor('#CCCCCC')

# --- 底部标注: 关键公式 ---
formula_text = (
    r'$X^{(i)} = [x_{i+1}, x_{i+2}, ..., x_{i+W}] \in \mathbb{R}^{W \times 17}$'
    '    |    '
    r'$y^{(k)} = \widetilde{RUL}^{(k+W-1)}$'
    '    |    '
    r'$W = 30$'
)
ax_top.text(95, -1.3, formula_text, ha='center', va='center',
            fontsize=11, color='#333333',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                      edgecolor='#CCCCCC', alpha=0.9))

# ========================
# 下子图: 不同发动机样本数对比
# ========================
ax_bot = fig.add_subplot(gs[1])
eng_ids = [1, 2, 91]
lifes = [engine_sample_counts[e]['life'] for e in eng_ids]
samples = [engine_sample_counts[e]['samples'] for e in eng_ids]
labels = ['Engine #1\n(192 cycles)', 'Engine #2\n(287 cycles)', 'Engine #91\n(135 cycles)']
colors_bar = ['#4C72B0', '#55A868', '#DD8452']

x = np.arange(len(eng_ids))
width = 0.35

# 柱状图: 寿命 vs 样本数
bars_life = ax_bot.bar(x - width/2, lifes, width, color='#B0BEC5',
                        edgecolor='white', linewidth=0.8, label='总寿命 (cycles)')
bars_samples = ax_bot.bar(x + width/2, samples, width, color=colors_bar,
                           edgecolor='white', linewidth=0.8, label='窗口样本数 (个)')

# 数值标签
for i, (life, samp) in enumerate(zip(lifes, samples)):
    ax_bot.text(i - width/2, life + 4, str(life), ha='center', fontsize=12,
                fontweight='bold', color='#555555')
    ax_bot.text(i + width/2, samp + 4, f'{samp}\n({life - WINDOW_SIZE + 1})',
                ha='center', fontsize=12, fontweight='bold', color='#222222')

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(labels, fontsize=12)
ax_bot.set_ylabel('数量', fontsize=13, fontweight='bold')
ax_bot.set_title('不同发动机寿命 → 不同窗口样本数量',
                 fontsize=14, fontweight='bold', pad=10)
ax_bot.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax_bot.set_ylim(0, max(lifes) * 1.25)

# 添加说明文字
ax_bot.text(0.5, -0.25,
            f'样本数 = max(0, 寿命 − W + 1)  |  W = {WINDOW_SIZE}  |  '
            f'引擎总样本数: {sum(samples)} 个 (FD001 训练集)',
            transform=ax_bot.transAxes, ha='center', va='top',
            fontsize=11, color='#666666', style='italic')

# ========================
# 保存
# ========================
out_path = os.path.join(FIG_DIR, 'ch2_sliding_window_demo.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\n✅ 图片已保存: {out_path}")
print("=" * 60)
