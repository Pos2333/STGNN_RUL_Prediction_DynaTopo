# ============================================================
# plot_ch2_rul_clip.py
# 第2章 RUL 标签截断前后对比图 —— 3台发动机 × 原始 vs 截断
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Seaborn 学术风格 ===
sns.set_theme(style='whitegrid', context='paper', font_scale=1.15,
              rc={'axes.edgecolor':'0.15','grid.alpha':0.25,
                  'figure.facecolor':'white','axes.facecolor':'#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
from configs.config import RUL_CLIP_MAX, RANDOM_SEED
np.random.seed(RANDOM_SEED)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# === 选取代表性发动机（与 plot_ch2_sensor_trends.py 保持一致） ===
SELECTED_ENGINES = [2, 1, 91]
ENGINE_LABELS = {
    2:  'Engine #2\n(长寿命, 287 cycles)',
    1:  'Engine #1\n(中等寿命, 192 cycles)',
    91: 'Engine #91\n(短寿命, 135 cycles)',
}

# === Nature 配色 ===
ORIGINAL_COLOR = '#7F8C8D'  # 灰色: 原始线性 RUL
CLIPPED_COLOR  = '#C44E52'  # Nature Red: 截断后 RUL
THRESHOLD_COLOR = '#4C72B0'  # Nature Blue: R_max=125 参考线
FILL_COLOR = '#EAEAF2'       # 浅灰: 截断区域填充

print("=" * 60)
print("  第2章: RUL 标签截断前后对比图")
print("=" * 60)

# === 加载 FD001 训练集原始数据 ===
RAW_DIR = os.path.join(PROJ_ROOT, 'data', 'raw')
df = pd.read_csv(os.path.join(RAW_DIR, 'train_FD001.txt'), sep=r'\s+', header=None)
print(f"原始数据加载完成: {df.shape}")

# === 为每台发动机提取生命周期信息 ===
engines_info = {}
for eng_id in SELECTED_ENGINES:
    eng_df = df[df[0] == eng_id]
    cycles = eng_df[1].values.astype(int)
    life = len(cycles)
    engines_info[eng_id] = {'life': life, 'cycles': np.arange(1, life + 1)}
    print(f"  Engine #{eng_id}: {life} cycles")

# ======================== 主图: 1×3 布局 (独立 Y 轴) ========================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
# 不使用 sharey —— 三台发动机 RUL 范围差异大 (286/191/134)，独立 Y 轴更清晰
fig.subplots_adjust(wspace=0.25, top=0.87, bottom=0.18, left=0.06, right=0.98)

for col, eng_id in enumerate(SELECTED_ENGINES):
    ax = axes[col]
    info = engines_info[eng_id]
    life = info['life']
    cycles = info['cycles']
    y_max = max(info['life'] - 1, RUL_CLIP_MAX)  # 该发动机的 Y 轴上限

    # ---- 原始 RUL 标签: 线性递减 ----
    original_rul = life - cycles  # [life-1, life-2, ..., 1, 0]
    ax.plot(cycles, original_rul, color=ORIGINAL_COLOR, linewidth=2.0,
            linestyle='--', alpha=0.8, label='原始 RUL ($RUL = T - t$)')

    # ---- 截断后 RUL 标签: 分段线性 ----
    clipped_rul = np.clip(original_rul, 0, RUL_CLIP_MAX)
    ax.plot(cycles, clipped_rul, color=CLIPPED_COLOR, linewidth=2.5,
            linestyle='-', alpha=0.95, label=f'截断 RUL ($R_{{max}}$ = {RUL_CLIP_MAX})')

    # ---- RUL_CLIP_MAX 水平参考线 ----
    ax.axhline(y=RUL_CLIP_MAX, color=THRESHOLD_COLOR, linewidth=1.2,
               linestyle=':', alpha=0.7)
    ax.text(life * 0.03, RUL_CLIP_MAX + 3, f'$R_{{max}}$ = {RUL_CLIP_MAX}',
            fontsize=9, color=THRESHOLD_COLOR, fontweight='bold', va='bottom')

    # ---- 截断区域半透明填充 ----
    mask_clipped = original_rul > RUL_CLIP_MAX
    if mask_clipped.any():
        clip_indices = np.where(mask_clipped)[0]
        ax.fill_between(cycles[clip_indices], original_rul[clip_indices],
                        RUL_CLIP_MAX, color=FILL_COLOR, alpha=0.4)

    # ---- 标注交拐点 ----
    if life > RUL_CLIP_MAX:
        knee_cycle = life - RUL_CLIP_MAX
        ax.plot(knee_cycle, RUL_CLIP_MAX, 'o', color=CLIPPED_COLOR,
                markersize=9, markeredgecolor='white', markeredgewidth=2.0,
                zorder=5)
        ax.annotate(f'截断点\n({knee_cycle}, {RUL_CLIP_MAX})',
                    xy=(knee_cycle, RUL_CLIP_MAX),
                    xytext=(knee_cycle - 30, RUL_CLIP_MAX + 30),
                    fontsize=8.5, color=CLIPPED_COLOR, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=CLIPPED_COLOR,
                                    lw=1.2, connectionstyle='arc3,rad=0.3'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=CLIPPED_COLOR, alpha=0.8))

    # ---- 标注原始 RUL 最大值 ----
    ax.text(life * 0.50, original_rul[0] - y_max * 0.06,
            f'原始最大值 = {original_rul[0]}',
            fontsize=8.5, color=ORIGINAL_COLOR, fontstyle='italic',
            ha='center')

    # ---- 坐标轴标注 ----
    ax.set_xlabel('运行周期 (Cycle)', fontsize=11, fontweight='bold')
    ax.set_ylabel('剩余使用寿命 (RUL)', fontsize=11, fontweight='bold')
    ax.set_title(ENGINE_LABELS[eng_id], fontsize=10.5, fontweight='bold',
                 color='#2C3E50', pad=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(-life * 0.02, life * 1.04)
    ax.set_ylim(-y_max * 0.03, y_max * 1.10)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85)
    ax.grid(True, alpha=0.2, linewidth=0.4)

# === 全局标题 ===
fig.suptitle('RUL 标签截断前后对比 (FD001, $R_{max}$ = 125)',
             fontsize=15, fontweight='bold', y=0.985)

# === 底部说明文字 ===
fig.text(0.5, 0.025,
         '— 原始线性 RUL: $RUL = T - t$　|　— 截断 RUL: $RUL = \\min(T - t, 125)$　|　灰色区域 = 被截断的寿命早期段',
         ha='center', fontsize=9.5, color='#7F8C8D',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f6fa',
                   edgecolor='#bdc3c7', alpha=0.85))

out_path = os.path.join(FIG_DIR, 'ch2_rul_clip_comparison.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\n✅ 图已保存: {out_path}")

# ======================== 补充: 单台大图 (Engine #2, 最直观) ========================
eng_id = 2
info = engines_info[eng_id]
life = info['life']
cycles = info['cycles']
original_rul = life - cycles
clipped_rul = np.clip(original_rul, 0, RUL_CLIP_MAX)

fig2, ax2 = plt.subplots(figsize=(10, 5.5))

# 原始 RUL
ax2.plot(cycles, original_rul, color=ORIGINAL_COLOR, linewidth=2.5,
         linestyle='--', alpha=0.7, label='原始 RUL (线性递减)')
# 截断 RUL
ax2.plot(cycles, clipped_rul, color=CLIPPED_COLOR, linewidth=3.0,
         linestyle='-', alpha=0.95, label=f'截断 RUL ($R_{{max}}$ = {RUL_CLIP_MAX})')
# R_max 参考线
ax2.axhline(y=RUL_CLIP_MAX, color=THRESHOLD_COLOR, linewidth=1.5,
            linestyle=':', alpha=0.6)

# 填充截断区域
mask_clipped = original_rul > RUL_CLIP_MAX
if mask_clipped.any():
    clip_start = np.where(mask_clipped)[0]
    ax2.fill_between(cycles[clip_start], original_rul[clip_start],
                     RUL_CLIP_MAX, color=FILL_COLOR, alpha=0.35,
                     label=f'截断区间 (RUL > {RUL_CLIP_MAX})')

# 交拐点
knee_cycle = life - RUL_CLIP_MAX
ax2.axvline(x=knee_cycle, color=THRESHOLD_COLOR, linewidth=1.2,
            linestyle=':', alpha=0.5)
ax2.plot(knee_cycle, RUL_CLIP_MAX, 'o', color=CLIPPED_COLOR, markersize=12,
         markeredgecolor='white', markeredgewidth=2.5, zorder=5)
ax2.annotate(f'截断点\n(Cycle = {knee_cycle}, RUL = {RUL_CLIP_MAX})',
             xy=(knee_cycle, RUL_CLIP_MAX),
             xytext=(knee_cycle - 55, RUL_CLIP_MAX + 35),
             fontsize=10.5, color=CLIPPED_COLOR, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=CLIPPED_COLOR,
                             lw=1.8, connectionstyle='arc3,rad=0.3'),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=CLIPPED_COLOR, alpha=0.85))

# 标注早期阶段
ax2.annotate('健康/早期退化阶段\n(传感器变化不显著,\n统一标为 RUL=125)',
             xy=(knee_cycle // 2, RUL_CLIP_MAX + 10),
             fontsize=9.5, color='#2C3E50', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                       edgecolor='#F9A825', alpha=0.8))

# 标注退化阶段
ax2.annotate('明显退化阶段\n(RUL 从 125\n线性递减至 0)',
             xy=(knee_cycle + (life - knee_cycle) // 2, 50),
             fontsize=9.5, color='#2C3E50', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE',
                       edgecolor=CLIPPED_COLOR, alpha=0.8))

ax2.set_xlabel('运行周期 (Cycle)', fontsize=12, fontweight='bold')
ax2.set_ylabel('剩余使用寿命 (RUL)', fontsize=12, fontweight='bold')
ax2.set_title(f'RUL 标签截断机制详解 —— Engine #2 (FD001, 总寿命 {life} cycles)',
              fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=9.5, loc='lower left', framealpha=0.9, ncol=2)
ax2.set_xlim(0, life * 1.03)
ax2.set_ylim(-5, original_rul[0] * 1.08)
ax2.tick_params(labelsize=10)
ax2.grid(True, alpha=0.15, linewidth=0.4)

plt.tight_layout()
out_path2 = os.path.join(FIG_DIR, 'ch2_rul_clip_detail.png')
fig2.savefig(out_path2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"✅ 详解图已保存: {out_path2}")

print("\n✅ 第2章 RUL 标签截断对比图全部完成!")
