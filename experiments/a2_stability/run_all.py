# ============================================================
# experiments/a2_stability/run_all.py
# 一键跑完所有 preset × seed 组合
# ============================================================
# 用法:
#   python experiments/a2_stability/run_all.py
#
# 组合数量由 config_seeds.py 动态决定：
#   当前为 5 个预设 × 5 个种子 = 25 次训练。
# 每次训练最多 200 epoch（含早停），预计总耗时取决于 GPU。
# ============================================================

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_seeds import SEEDS, PRESETS, LOG_DIR

TRAIN_SCRIPT = os.path.join(ROOT, 'experiments', 'a2_stability', 'train_multi_seed.py')


def main():
    total = len(PRESETS) * len(SEEDS)
    print("=" * 60)
    print(f"  多 seed 稳定性实验：{len(PRESETS)} 预设 × {len(SEEDS)} 种子 = {total} 次训练")
    print("=" * 60)

    count = 0
    skipped = 0
    for preset in PRESETS:
        for seed in SEEDS:
            log_path = os.path.join(ROOT, LOG_DIR, f'{preset}_seed{seed}.json')
            # 断点续跑：已有日志则跳过（云端 SSH 断线后重跑可自动续上）
            if os.path.exists(log_path):
                print(f"  ⏭️  跳过（已有日志）: preset={preset}, seed={seed}")
                skipped += 1
                continue
            count += 1
            print(f"\n{'─'*60}")
            print(f"  [{count}/{total}] preset={preset}, seed={seed}")
            print(f"{'─'*60}")

            cmd = [sys.executable, TRAIN_SCRIPT, '--preset', preset, '--seed', str(seed)]
            ret = subprocess.run(cmd, cwd=ROOT)

            if ret.returncode != 0:
                print(f"\n  ❌ 训练失败: preset={preset}, seed={seed}")
                print(f"  继续下一个组合...")
                continue

    print(f"\n{'='*60}")
    print(f"  ✅ 全部完成！本次运行 {count} 次，跳过已有 {skipped} 次（共 {total} 次）")
    print(f"  运行分析脚本: python experiments/a2_stability/analyze.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
