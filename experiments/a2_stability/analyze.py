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
    for f in glob.glob(os.path.join(log_dir, '*.json')):
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

    print(f"\n  {'模型':<8} {'种子数':<6} {'test RMSE':<22} {'test NASA':<22} {'val RMSE':<22}")
    print("  " + "─" * 66)

    summary = {}
    for preset in PRESETS:
        seeds_data = results[preset]
        n = len(seeds_data)
        if n == 0:
            print(f"  {preset:<8} {'0':<6} 无数据")
            continue

        test_rmses = [d['test_rmse'] for d in seeds_data.values()]
        test_nasas = [d['test_nasa_score'] for d in seeds_data.values()]
        val_rmses = [d['val_rmse'] for d in seeds_data.values()]

        rmse_mean, rmse_std = np.mean(test_rmses), np.std(test_rmses)
        nasa_mean, nasa_std = np.mean(test_nasas), np.std(test_nasas)
        val_mean, val_std = np.mean(val_rmses), np.std(val_rmses)

        print(f"  {preset:<8} {n:<6} "
              f"{rmse_mean:>6.2f} ± {rmse_std:<12.2f} "
              f"{nasa_mean:>7.1f} ± {nasa_std:<10.1f} "
              f"{val_mean:>6.2f} ± {val_std:<12.2f}")

        summary[preset] = {
            'n': n,
            'test_rmse_mean': float(rmse_mean), 'test_rmse_std': float(rmse_std),
            'test_nasa_mean': float(nasa_mean), 'test_nasa_std': float(nasa_std),
            'val_rmse_mean': float(val_mean), 'val_rmse_std': float(val_std),
        }

    # 保存汇总
    out_path = os.path.join(ROOT, LOG_DIR, 'summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📝 汇总已保存 → {out_path}")
    return summary


def plot_box(summary):
    """绘制箱线图对比 A1/A2 的离散程度"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 不可用，跳过绘图")
        return

    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 重新加载原始数据用于箱线图
    results = load_all_results()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [(axes[0], 'test_rmse', 'Test RMSE (越小越好)'),
                           (axes[1], 'test_nasa_score', 'Test NASA Score (越小越好)')]:
        data = [[d[key] for d in results[p].values()] for p in PRESETS]
        ax.boxplot(data, labels=PRESETS, showmeans=True)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(key)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('多 seed 稳定性对比：A1 vs A2（修正 softmax 后）', fontsize=15, fontweight='bold')
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
