# ============================================================
# plot_task5_transfer.py
# 任务5 跨工况迁移结果可视化 —— 目标域测试集 RMSE & NASA Score 对比
#
# 数据来源: logs/dynatopo/eval_cross_condition.json
#           （由 evaluate_2_dynatopo.py --preset all 生成）
# 输出:
#   figures/transfer_rmse_by_model.png        —— RMSE 分面柱状图（4 模型 × 3 目标域 × 4 方式）
#   figures/transfer_nasa_by_model.png        —— NASA Score 分面柱状图（log 尺度）
#   figures/transfer_uda_rmse_comparison.png  —— UDA 场景 RMSE 对比（突出动态图优势）
#   figures/transfer_uda_improvement.png      —— UDA 相对无迁移的 RMSE 降幅
#
# 风格: seaborn 学术风格 + 中文标注（与 notebooks/ 其他 plot_*.py 一致）
#
# 运行:
#   conda activate rul_env
#   python notebooks/plot_task5_transfer.py
# ============================================================
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='paper', font_scale=1.05,
              rc={'axes.edgecolor': '0.15', 'grid.alpha': 0.2,
                  'figure.facecolor': 'white', 'axes.facecolor': '#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

RESULT_PATH = os.path.join(ROOT, 'logs', 'dynatopo', 'eval_cross_condition.json')
with open(RESULT_PATH, encoding='utf-8') as f:
    data = json.load(f)

PRESETS = ['static', 'A1B1', 'A2B1', 'A2B2']
TARGETS = ['FD002', 'FD003', 'FD004']
MODES = [('no_transfer', '无迁移'),
         ('semi', 'LMMD半监督'),
         ('none', '无自适应微调'),
         ('uda', '无监督UDA')]

# 每个模型一个颜色（同组内不同模型用颜色区分，突出模型对比）
MODEL_PALETTE = {'static': '#7F8C8D',   # 灰
                 'A1B1': '#1F77B4',    # 蓝
                 'A2B1': '#2CA02C',    # 绿
                 'A2B2': '#D62728'}    # 红

MODEL_NAMES = {'static': '静态基线 (STGNN-静态)',
               'A1B1': 'A1B1 (相似度×特征融合)',
               'A2B1': 'A2B1 (注意力×特征融合)',
               'A2B2': 'A2B2 (注意力×拓扑融合)'}

MODE_LABELS = [m[1] for m in MODES]

# ---- 整理为 DataFrame ----
rows = []
for k, r in data.items():
    for mkey, mlabel in MODES:
        rmse = r.get(f'{mkey}_rmse')
        score = r.get(f'{mkey}_score')
        if rmse is None or score is None:   # 缺模型时跳过（保持健壮）
            continue
        rows.append({'模型': r['preset'], '目标域': r['target'], '方式': mlabel,
                     'RMSE': float(rmse), 'NASA Score': float(score)})
df = pd.DataFrame(rows)
print(f"📊 数据点: {len(df)} 条（4 模型 × 3 目标域 × 4 方式 = 48）")


def draw(metric, ylabel, log=False, fname=None):
    """
    柱状图：col = 迁移方式（3 子图），x = 目标域（3 组），hue = 模型（每组 4 柱）
    —— 以「同一工况、同一迁移方式、不同模型」为一组，突出不同模型之间的比较
    """
    g = sns.catplot(data=df, x='目标域', y=metric, hue='模型', kind='bar',
                    col='方式', col_order=MODE_LABELS, col_wrap=2,
                    palette=MODEL_PALETTE,
                    height=3.4, aspect=1.15, legend_out=False,
                    edgecolor='0.3', linewidth=0.6)
    g.set_axis_labels('目标域（测试集）', ylabel)
    for ax, mode in zip(g.axes.flat, MODE_LABELS):
        ax.set_title(f'迁移方式：{mode}', fontsize=12, fontweight='bold')
        if log:
            ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=10)
        # 柱顶标注数值
        for container in ax.containers:
            labels = []
            for v in container.datavalues:
                if v is None:
                    labels.append('')
                elif log:
                    labels.append(f'{v:.0f}')
                else:
                    labels.append(f'{v:.1f}')
            ax.bar_label(container, labels=labels, fontsize=6.5, padding=2)
    # 图例展示模型中文名
    g.add_legend(title='模型', fontsize=9)
    for t, preset in zip(g._legend.get_texts(), PRESETS):
        t.set_text(MODEL_NAMES[preset])
    g.fig.suptitle(f'跨工况迁移对比：同一工况、同一方式下不同模型的 {ylabel}（越低越好）',
                   fontsize=14, fontweight='bold', y=1.03)
    out = os.path.join(FIG_DIR, fname)
    g.fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 已保存 → {out}")


# ---- 图1: RMSE ----
draw('RMSE', 'RMSE', log=False, fname='transfer_rmse_by_model.png')

# ---- 图2: NASA Score（log 尺度，跨度从 10^2 到 10^6）----
draw('NASA Score', 'NASA Score', log=True, fname='transfer_nasa_by_model.png')


def draw_uda_highlight():
    """图3+图4：突出 UDA 无监督场景下动态图相对静态基线的优势（重大发现）"""
    sub = df[df['方式'] == '无监督UDA'].copy()

    # ---- 图3: UDA 场景 RMSE 对比 ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=sub, x='目标域', y='RMSE', hue='模型',
                palette=MODEL_PALETTE, ax=ax, edgecolor='0.3', linewidth=0.6)
    ax.set_title('无监督 UDA 跨工况迁移：各模型目标域测试 RMSE（越低越好）',
                 fontsize=12.5, fontweight='bold')
    ax.set_xlabel('目标域（测试集）')
    ax.set_ylabel('RMSE')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f', fontsize=7.5, padding=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [MODEL_NAMES[p] for p in PRESETS], title='模型', fontsize=8)
    out = os.path.join(FIG_DIR, 'transfer_uda_rmse_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 已保存 → {out}")
    plt.close(fig)

    # ---- 图4: UDA 相对无迁移的 RMSE 降幅（正=UDA 有效，负=失效）----
    imp = []
    for k, r in data.items():
        if r.get('no_transfer_rmse') is None or r.get('uda_rmse') is None:
            continue
        imp.append({'模型': r['preset'], '目标域': r['target'],
                    'RMSE降幅': float(r['no_transfer_rmse']) - float(r['uda_rmse'])})
    imp_df = pd.DataFrame(imp)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=imp_df, x='目标域', y='RMSE降幅', hue='模型',
                palette=MODEL_PALETTE, ax=ax, edgecolor='0.3', linewidth=0.6)
    ax.axhline(0, color='0.5', lw=1)
    ax.set_title('UDA 相对无迁移的 RMSE 降幅（正=UDA 有效，负=失效）',
                 fontsize=12.5, fontweight='bold')
    ax.set_xlabel('目标域（测试集）')
    ax.set_ylabel('RMSE 降幅 (无迁移 - UDA)')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f', fontsize=7.5, padding=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [MODEL_NAMES[p] for p in PRESETS], title='模型', fontsize=8)
    out = os.path.join(FIG_DIR, 'transfer_uda_improvement.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 已保存 → {out}")
    plt.close(fig)


draw_uda_highlight()

print("\n✅ 全部图表生成完毕！")
