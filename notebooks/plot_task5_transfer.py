# ============================================================
# plot_task5_transfer.py
# 任务5 跨工况迁移结果可视化 —— 验证集 + 测试集指标对比 & UDA 降幅
#
# 数据来源:
#   logs/dynatopo/eval_cross_condition.json   —— 目标域测试集 RMSE & NASA
#   logs/dynatopo/eval_transfer_val.json      —— 目标域验证集 RMSE & NASA（重估）
# 输出:
#   figures/transfer_val_grid.png         —— 验证集指标网格（RMSE+NASA × 3 方式）
#   figures/transfer_test_grid.png        —— 测试集指标网格（RMSE+NASA × 3 方式）
#   figures/transfer_uda_improvement.png  —— UDA 相对无迁移的 4 指标降幅
#   figures/transfer_uda_improvement_heatmap.png —— 降幅热力图（2×2 面板）
#
# 运行:
#   python notebooks/plot_task5_transfer.py
# ============================================================
import os
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

with open(os.path.join(ROOT, 'logs', 'dynatopo', 'eval_cross_condition.json'),
          encoding='utf-8') as f:
    test_data = json.load(f)
with open(os.path.join(ROOT, 'logs', 'dynatopo', 'eval_transfer_val.json'),
          encoding='utf-8') as f:
    val_data = json.load(f)

PRESETS = ['static', 'A1B1', 'A2B1', 'A2B2']
TARGETS = ['FD002', 'FD003', 'FD004']

# 仅考虑三种迁移方式（忽略"无自适应微调"），从左到右：监督式 → 无迁移 → 无监督
MODES = [('lmmd_semi', '监督式域自适应'),
         ('no_transfer', '无迁移'),
         ('lmmd_uda', '无监督域自适应')]

# 字段映射：mode -> 各指标在 json 中的 key
VAL_RMSE_KEY = {'no_transfer': 'no_transfer_val_rmse',
                'lmmd_semi': 'lmmd_semi_val_rmse',
                'lmmd_uda': 'lmmd_uda_val_rmse'}
VAL_NASA_KEY = {'no_transfer': 'no_transfer_val_nasa',
                'lmmd_semi': 'lmmd_semi_val_nasa',
                'lmmd_uda': 'lmmd_uda_val_nasa'}
TEST_RMSE_KEY = {'no_transfer': 'no_transfer_rmse',
                 'lmmd_semi': 'semi_rmse',
                 'lmmd_uda': 'uda_rmse'}
TEST_NASA_KEY = {'no_transfer': 'no_transfer_score',
                 'lmmd_semi': 'semi_score',
                 'lmmd_uda': 'uda_score'}

MODEL_PALETTE = {'static': '#7F8C8D', 'A1B1': '#1F77B4',
                 'A2B1': '#2CA02C', 'A2B2': '#D62728'}
MODEL_NAMES = {'static': '静态基线 (STGNN-静态)',
               'A1B1': 'A1B1 (相似度×特征融合)',
               'A2B1': 'A2B1 (注意力×特征融合)',
               'A2B2': 'A2B2 (注意力×拓扑融合)'}
MODE_LABELS = [m[1] for m in MODES]

# ---- 整理为 DataFrame ----
rows = []
for p in PRESETS:
    for t in TARGETS:
        te = test_data[f'{p}_{t}']
        ve = val_data[f'{p}_{t}']
        for mkey, mlabel in MODES:
            rows.append({
                '模型': p, '目标域': t, '方式': mlabel,
                '验证集RMSE': float(ve[VAL_RMSE_KEY[mkey]]),
                '验证集NASA': float(ve[VAL_NASA_KEY[mkey]]),
                '测试集RMSE': float(te[TEST_RMSE_KEY[mkey]]),
                '测试集NASA': float(te[TEST_NASA_KEY[mkey]]),
            })
df = pd.DataFrame(rows)
print(f"📊 数据点: {len(df)} 条（4 模型 × 3 目标域 × 3 方式 = 36）")


def _style_legend(g):
    g.add_legend(title='模型', fontsize=9)
    for t, preset in zip(g._legend.get_texts(), PRESETS):
        t.set_text(MODEL_NAMES[preset])


def draw_set_grid(metric_cols, fname, suptitle=''):
    """验证集或测试集指标网格：row=指标(RMSE/NASA)，col=方式（监督式→无迁移→UDA）"""
    long = df.melt(id_vars=['模型', '目标域', '方式'], value_vars=metric_cols,
                   var_name='指标', value_name='值')
    g = sns.catplot(data=long, x='目标域', y='值', hue='模型',
                    row='指标', col='方式', kind='bar',
                    row_order=metric_cols, col_order=MODE_LABELS,
                    palette=MODEL_PALETTE, sharey='row',
                    height=3.2, aspect=1.35, legend_out=False,
                    edgecolor='0.3', linewidth=0.6)
    # NASA 行用 log 尺度（量级跨 10^3~10^7）
    for row_i, metric in enumerate(metric_cols):
        if 'NASA' in metric:
            for ax in g.axes[row_i]:
                ax.set_yscale('log')
    g.set_axis_labels('目标域', '指标值')
    for ax in g.axes.flat:
        for container in ax.containers:
            labels = []
            for v in container.datavalues:
                if v is None or v <= 0:
                    labels.append('')
                elif ax.get_yscale() == 'log':
                    labels.append(f'{v:.1e}')
                else:
                    labels.append(f'{v:.1f}')
            ax.bar_label(container, labels=labels, fontsize=6, padding=1.5)
    _style_legend(g)
    g.fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    out = os.path.join(FIG_DIR, fname)
    g.fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 已保存 → {out}")


def draw_improvement():
    """UDA 相对无迁移的 4 指标降幅（正=UDA 有效）"""
    imp_rows = []
    for p in PRESETS:
        for t in TARGETS:
            te = test_data[f'{p}_{t}']
            ve = val_data[f'{p}_{t}']
            imp_rows.append({
                '模型': p, '目标域': t,
                '验证集RMSE降幅': ve['no_transfer_val_rmse'] - ve['lmmd_uda_val_rmse'],
                '验证集NASA降幅': ve['no_transfer_val_nasa'] - ve['lmmd_uda_val_nasa'],
                '测试集RMSE降幅': te['no_transfer_rmse'] - te['uda_rmse'],
                '测试集NASA降幅': te['no_transfer_score'] - te['uda_score'],
            })
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(FIG_DIR, 'transfer_uda_improvement.csv'),
               index=False, encoding='utf-8-sig')

    metric_cols = ['验证集RMSE降幅', '验证集NASA降幅', '测试集RMSE降幅', '测试集NASA降幅']
    long = imp.melt(id_vars=['模型', '目标域'], value_vars=metric_cols,
                    var_name='指标', value_name='降幅')
    g = sns.catplot(data=long, x='目标域', y='降幅', hue='模型',
                    row='指标', kind='bar', row_order=metric_cols,
                    palette=MODEL_PALETTE, sharey=False,
                    height=2.9, aspect=1.5, legend_out=False,
                    edgecolor='0.3', linewidth=0.6)
    # 每子图加 0 参考线
    for ax in g.axes.flat:
        ax.axhline(0, color='0.5', lw=1)
        for container in ax.containers:
            labels = []
            for v in container.datavalues:
                labels.append(f'{v:.0f}' if v is not None else '')
            ax.bar_label(container, labels=labels, fontsize=6, padding=1.5)
    _style_legend(g)
    g.fig.suptitle('UDA 相对无迁移的指标降幅（正=UDA 有效，负=失效）',
                   fontsize=14, fontweight='bold', y=1.02)
    out = os.path.join(FIG_DIR, 'transfer_uda_improvement.png')
    g.fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 已保存 → {out}")
    return imp


def draw_improvement_heatmap(imp):
    """UDA 相对无迁移的降幅热力图：正值红→黄→绿（大=绿），负值一律中性灰（保留数值）"""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    metric_cols = ['验证集RMSE降幅', '验证集NASA降幅', '测试集RMSE降幅', '测试集NASA降幅']
    fmts = ['.1f', '.1e', '.1f', '.1e']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, col, fmt in zip(axes.flat, metric_cols, fmts):
        pivot = imp.pivot(index='模型', columns='目标域', values=col)
        pivot = pivot.reindex(index=PRESETS, columns=TARGETS)

        vmax = max(float(pivot.values.max()), 1e-6)
        # 色图：首色=中性灰（负值），后续=红→黄→绿（正值 0→vmax）
        pos_colors = sns.color_palette('RdYlGn', 256)  # 红→黄→绿
        cmap = ListedColormap(['#999999'] + list(pos_colors))
        bounds = [-np.inf] + list(np.linspace(0.0, vmax, 257))
        norm = BoundaryNorm(bounds, cmap.N)

        sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt, cmap=cmap, norm=norm,
                    cbar=False, linewidths=0.6, linecolor='white')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('目标域')
        ax.set_ylabel('模型')
    fig.suptitle('UDA 相对无迁移的降幅热力图（绿=大改善，黄=中，红=小改善，灰=失效）',
                 fontsize=14.5, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, 'transfer_uda_improvement_heatmap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 已保存 → {out}")
    plt.close(fig)


# ---- 图1: 验证集指标网格（RMSE + NASA）----
draw_set_grid(['验证集RMSE', '验证集NASA'],
              fname='transfer_val_grid.png',
              suptitle='目标域验证集指标跨工况对比（越低越好，NASA 为 log 尺度）')

# ---- 图2: 测试集指标网格（RMSE + NASA）----
draw_set_grid(['测试集RMSE', '测试集NASA'],
              fname='transfer_test_grid.png',
              suptitle='目标域测试集指标跨工况对比（越低越好，NASA 为 log 尺度）')

# ---- 图3: UDA 相对无迁移的降幅 ----
imp = draw_improvement()

# ---- 图4: 降幅热力图 ----
draw_improvement_heatmap(imp)

print("\n✅ 全部图表生成完毕！")
