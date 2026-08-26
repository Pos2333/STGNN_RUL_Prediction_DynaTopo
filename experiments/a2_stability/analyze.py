# ============================================================
# experiments/a2_stability/analyze.py
# 汇总多 seed 实验的 mean ± std，并绘制箱线图
# ============================================================
# 用法:
#   python experiments/a2_stability/analyze.py
# ============================================================

import os
import sys
import json
import glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_seeds import SEEDS, PRESETS, LOG_DIR


def load_all_results():
    """加载所有 (preset, seed) 的日志"""
    results = {preset: {} for preset in PRESETS}
    log_dir = os.path.join(ROOT, LOG_DIR)
    # 只匹配 *_seed*.json，跳过 summary.json 等汇总文件
    for f in glob.glob(os.path.join(log_dir, '*_seed*.json')):
        with open(f) as fp:
            data = json.load(fp)
        preset = data['preset']
        seed = data['seed']
        if preset in results:
            results[preset][seed] = data
    return results


def summarize(results):
    """计算每个预设的 mean ± std"""
    print("=" * 70)
    print("  多 seed 稳定性分析结果")
    print("=" * 70)

    print(f"\n  {'模型':<8} {'种子数':<6} {'val RMSE':<22} {'val NASA':<22} {'test RMSE':<22} {'test NASA':<22} {'参数量':<12}")
    print("  " + "─" * 110)

    summary = {}
    for preset in PRESETS:
        seeds_data = results[preset]
        n = len(seeds_data)
        if n == 0:
            print(f"  {preset:<8} {'0':<6} 无数据")
            continue

        val_rmses = [d['val_rmse'] for d in seeds_data.values()]
        val_nasas = [d['val_nasa_score'] for d in seeds_data.values()]
        test_rmses = [d['test_rmse'] for d in seeds_data.values()]
        test_nasas = [d['test_nasa_score'] for d in seeds_data.values()]
        params = next(iter(seeds_data.values())).get('params', 0)

        val_mean, val_std = np.mean(val_rmses), np.std(val_rmses)
        val_nasa_mean, val_nasa_std = np.mean(val_nasas), np.std(val_nasas)
        rmse_mean, rmse_std = np.mean(test_rmses), np.std(test_rmses)
        nasa_mean, nasa_std = np.mean(test_nasas), np.std(test_nasas)

        print(f"  {preset:<8} {n:<6} "
              f"{val_mean:>6.2f} ± {val_std:<12.2f} "
              f"{val_nasa_mean:>8.1f} ± {val_nasa_std:<10.1f} "
              f"{rmse_mean:>6.2f} ± {rmse_std:<12.2f} "
              f"{nasa_mean:>7.1f} ± {nasa_std:<10.1f} "
              f"{params:>12,}")

        summary[preset] = {
            'n': n,
            'test_rmse_mean': float(rmse_mean), 'test_rmse_std': float(rmse_std),
            'test_nasa_mean': float(nasa_mean), 'test_nasa_std': float(nasa_std),
            'val_rmse_mean': float(val_mean), 'val_rmse_std': float(val_std),
            'val_nasa_mean': float(val_nasa_mean), 'val_nasa_std': float(val_nasa_std),
        }

    # 保存汇总
    out_path = os.path.join(ROOT, LOG_DIR, 'summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📝 汇总已保存 → {out_path}")
    return summary


def plot_box(summary):
    """绘制四指标箱线图（验证集/测试集的 RMSE 与 NASA Score），中文学术风格"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️  matplotlib / seaborn 不可用，跳过绘图")
        return

    # ---- 学术风格（与 notebooks 其他绘图一致）----
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.05,
                  rc={'axes.edgecolor': '0.15', 'grid.alpha': 0.2,
                      'figure.facecolor': 'white', 'axes.facecolor': '#fafafa'})
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ---- 模型配色与中文名 ----
    MODEL_PALETTE = {'static': '#7F8C8D',
                     'A1B1': '#1F77B4',
                     'A1B2': '#FF7F0E',
                     'A2B1': '#2CA02C',
                     'A2B2': '#D62728'}
    MODEL_NAMES = {'static': '静态基线',
                   'A1B1': 'A1B1\n(相似度×特征融合)',
                   'A1B2': 'A1B2\n(相似度×拓扑融合)',
                   'A2B1': 'A2B1\n(注意力×特征融合)',
                   'A2B2': 'A2B2\n(注意力×拓扑融合)'}

    # ---- 重新加载原始数据用于箱线图 ----
    results = load_all_results()

    # 四指标面板（2×2）：验证集与测试集的 RMSE / NASA Score
    panels = [
        ('val_rmse',        '验证集 RMSE',       'RMSE'),
        ('val_nasa_score',  '验证集 NASA Score', 'NASA Score'),
        ('test_rmse',       '测试集 RMSE',       'RMSE'),
        ('test_nasa_score', '测试集 NASA Score', 'NASA Score'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        data = [[d[key] for d in results[p].values()] for p in PRESETS]
        bp = ax.boxplot(
            data,
            showmeans=False,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.2),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        # 箱体填充颜色（对应模型配色）
        for patch, p in zip(bp['boxes'], PRESETS):
            patch.set_facecolor(MODEL_PALETTE[p])
            patch.set_alpha(0.55)
        # 叠加均值±标准差误差棒（与汇总表格严格一致：白菱形=均值，误差棒=±std）
        for i, p in enumerate(PRESETS):
            vals = np.array([d[key] for d in results[p].values()])
            ax.errorbar(i + 1, vals.mean(), yerr=vals.std(),
                        fmt='D', markersize=6,
                        markerfacecolor='white', markeredgecolor='black',
                        ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5,
                        zorder=10)
        ax.set_xticklabels([MODEL_NAMES[p] for p in PRESETS], fontsize=9)
        ax.set_title(title + '（越低越好）', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('多 seed 稳定性分析（箱体=四分位距，◇=均值±标准差）',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(ROOT, LOG_DIR, 'stability_boxplot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"📊 箱线图已保存 → {out}")


def main():
    results = load_all_results()
    if not any(results.values()):
        print("❌ 未找到任何实验日志，请先运行 train_multi_seed.py 或 run_all.py")
        return
    summary = summarize(results)
    plot_box(summary)

    # 结论提示
    print(f"\n{'='*70}")
    print("  解读提示")
    print(f"{'='*70}")
    print("  · 若 A2 的 std 明显大于 A1 的 std → A2 更不稳定")
    print("  · 若 A1 与 A2 的区间 (mean±std) 重叠 → 无显著差异")
    print("  · 若 A1 区间完全低于 A2 → A1 显著更优（需结合 NASA Score 判断）")


if __name__ == '__main__':
    main()
