import re
import csv
import numpy as np
from pathlib import Path

WORK_ROOT = Path("/home/srtp/ucf101_filter_work")
FUSION_LOG_DIR = WORK_ROOT / "fusion_logs"
FUSION_NPZ_DIR = WORK_ROOT / "fusion_scores" / "fused_211"
FILTER_SUMMARY = WORK_ROOT / "filter_summary" / "ucf101_filter_summary.csv"

def parse_accuracy_from_log(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r'Accuracy\s+([0-9.]+)%', text)
    if matches:
        return float(matches[-1])
    return None

def parse_accuracy_from_npz(path: Path):
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if "accuracy" in z.files:
        return float(z["accuracy"])
    return None

print("=" * 72)
print("1. Residual 单流结果（来自 filter_summary）")
print("=" * 72)

if FILTER_SUMMARY.exists():
    rows = list(csv.DictReader(FILTER_SUMMARY.open()))
    print("split, mode, cutoff, accuracy")
    for r in rows:
        print(f"{r['split']}, {r['mode']}, {r['cutoff']}, {r['accuracy']}")
else:
    print("未找到:", FILTER_SUMMARY)

print()
print("=" * 72)
print("2. 当前导出的单流测试结果（iframe / mv / residual）")
print("=" * 72)

single_results = []
for split in [1, 2, 3]:
    iframe_acc = parse_accuracy_from_log(FUSION_LOG_DIR / f"ucf101_split{split}_iframe_test.log")
    mv_acc = parse_accuracy_from_log(FUSION_LOG_DIR / f"ucf101_split{split}_mv_test.log")
    res_base_acc = parse_accuracy_from_log(FUSION_LOG_DIR / f"ucf101_split{split}_residual_baseline_test.log")
    res_filt_acc = parse_accuracy_from_log(FUSION_LOG_DIR / f"ucf101_split{split}_residual_fft035_test.log")

    single_results.append((split, iframe_acc, mv_acc, res_base_acc, res_filt_acc))

print("split, iframe, mv, residual_baseline, residual_fft035")
for row in single_results:
    print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}")

def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

iframe_mean = mean_or_none([r[1] for r in single_results])
mv_mean = mean_or_none([r[2] for r in single_results])
res_base_mean = mean_or_none([r[3] for r in single_results])
res_filt_mean = mean_or_none([r[4] for r in single_results])

print(f"mean, {iframe_mean}, {mv_mean}, {res_base_mean}, {res_filt_mean}")

print()
print("=" * 72)
print("3. 原论文固定权重 2:1:1 三流融合结果")
print("=" * 72)

fusion_results = []
for split in [1, 2, 3]:
    base_acc = parse_accuracy_from_npz(FUSION_NPZ_DIR / f"ucf101_split{split}_fusion_baseline_211.npz")
    filt_acc = parse_accuracy_from_npz(FUSION_NPZ_DIR / f"ucf101_split{split}_fusion_fft035_211.npz")
    delta = None if (base_acc is None or filt_acc is None) else (filt_acc - base_acc)
    fusion_results.append((split, base_acc, filt_acc, delta))

print("split, baseline_fusion_211, fft035_fusion_211, delta")
for row in fusion_results:
    print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}")

base_mean = mean_or_none([r[1] for r in fusion_results])
filt_mean = mean_or_none([r[2] for r in fusion_results])
delta_mean = None if (base_mean is None or filt_mean is None) else (filt_mean - base_mean)

print(f"mean, {base_mean}, {filt_mean}, {delta_mean}")

print()
print("=" * 72)
print("4. 结论")
print("=" * 72)

if res_base_mean is not None and res_filt_mean is not None:
    print(f"Residual 单流：fft035 相比 baseline 的平均变化 = {res_filt_mean - res_base_mean:.4f}")

if base_mean is not None and filt_mean is not None:
    print(f"固定权重 2:1:1 三流融合：fft035 相比 baseline 的平均变化 = {filt_mean - base_mean:.4f}")

print()
print("结果目录：")
print("单流日志目录:", FUSION_LOG_DIR)
print("融合 npz 目录:", FUSION_NPZ_DIR)
print("残差汇总 csv:", FILTER_SUMMARY)
