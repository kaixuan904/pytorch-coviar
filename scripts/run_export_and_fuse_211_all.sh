#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/srtp/ucf101_filter_work"
BASELINE_ROOT="/home/srtp/pytorch-coviar-ucf101"
DATA_ROOT="/home/srtp/data_fast/ucf101/mpeg4_videos"
LIST_DIR="${WORK_ROOT}/data/datalists"

OUT_DIR="${WORK_ROOT}/fusion_scores"
LOG_DIR="${WORK_ROOT}/fusion_logs"
FUSED_DIR="${OUT_DIR}/fused_211"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${FUSED_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate coviar

# -----------------------------
# 1. 生成 2:1:1 融合脚本
# -----------------------------
cat > "${WORK_ROOT}/fixed_fuse_npz.py" <<'PY'
import argparse
from pathlib import Path
import numpy as np

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    files = list(z.files)

    scores = z['scores'] if 'scores' in files else z[files[0]]
    labels = z['labels'] if 'labels' in files else None
    names = z['names'] if 'names' in files else None

    scores = np.array(scores)
    if scores.ndim == 3:
        scores = scores.squeeze(1)
    if scores.ndim != 2:
        raise ValueError(f"Unexpected score shape in {path}: {scores.shape}")

    return scores, labels, names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iframe', required=True)
    parser.add_argument('--mv', required=True)
    parser.add_argument('--residual', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--w_iframe', type=float, default=2.0)
    parser.add_argument('--w_mv', type=float, default=1.0)
    parser.add_argument('--w_residual', type=float, default=1.0)
    args = parser.parse_args()

    i_scores, i_labels, i_names = load_npz(args.iframe)
    m_scores, m_labels, m_names = load_npz(args.mv)
    r_scores, r_labels, r_names = load_npz(args.residual)

    if not (len(i_scores) == len(m_scores) == len(r_scores)):
        raise ValueError("Three streams have different sample counts.")

    if i_names is not None and m_names is not None and r_names is not None:
        if not (np.array_equal(i_names, m_names) and np.array_equal(i_names, r_names)):
            raise ValueError("Sample names are not aligned across streams.")

    labels = i_labels
    if labels is None and m_labels is not None:
        labels = m_labels
    if labels is None and r_labels is not None:
        labels = r_labels

    fused = (
        args.w_iframe * i_scores +
        args.w_mv * m_scores +
        args.w_residual * r_scores
    )

    pred = fused.argmax(axis=1)
    acc = None
    if labels is not None:
        labels = np.array(labels).astype(np.int64)
        acc = float((pred == labels).mean() * 100.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        scores=fused,
        labels=labels,
        names=i_names if i_names is not None else None,
        pred=pred,
        w_iframe=args.w_iframe,
        w_mv=args.w_mv,
        w_residual=args.w_residual,
        accuracy=acc if acc is not None else -1.0,
    )

    print(f"[DONE] saved fused npz -> {out}")
    if acc is not None:
        print(f"[ACC] {acc:.4f}")

if __name__ == '__main__':
    main()
PY

# -----------------------------
# 2. 导出单流 .npz
# -----------------------------
for split in 1 2 3; do
  TEST_LIST="${LIST_DIR}/ucf101_split${split}_test.txt"

  # iframe
  python "${WORK_ROOT}/test.py" \
    --data-name ucf101 \
    --data-root "${DATA_ROOT}" \
    --test-list "${TEST_LIST}" \
    --representation iframe \
    --arch resnet18 \
    --weights "${BASELINE_ROOT}/ucf101_split${split}_iframe_resnet18_iframe_model_best.pth.tar" \
    --save-scores "${OUT_DIR}/ucf101_split${split}_iframe.npz" \
    -j 8 \
    --gpus 0 \
    2>&1 | tee "${LOG_DIR}/ucf101_split${split}_iframe_test.log"

  # mv
  python "${WORK_ROOT}/test.py" \
    --data-name ucf101 \
    --data-root "${DATA_ROOT}" \
    --test-list "${TEST_LIST}" \
    --representation mv \
    --arch resnet18 \
    --weights "${BASELINE_ROOT}/ucf101_split${split}_mv_resnet18_mv_model_best.pth.tar" \
    --save-scores "${OUT_DIR}/ucf101_split${split}_mv.npz" \
    -j 8 \
    --gpus 0 \
    2>&1 | tee "${LOG_DIR}/ucf101_split${split}_mv_test.log"

  # residual baseline
  python "${WORK_ROOT}/test.py" \
    --data-name ucf101 \
    --data-root "${DATA_ROOT}" \
    --test-list "${TEST_LIST}" \
    --representation residual \
    --arch resnet18 \
    --weights "${BASELINE_ROOT}/ucf101_split${split}_residual_resnet18_residual_model_best.pth.tar" \
    --save-scores "${OUT_DIR}/ucf101_split${split}_residual_baseline.npz" \
    -j 8 \
    --gpus 0 \
    2>&1 | tee "${LOG_DIR}/ucf101_split${split}_residual_baseline_test.log"

  # residual filtered (fft035)
  FILTER_BEST="${WORK_ROOT}/filter_checkpoints/ucf101_split${split}_residual_fft035_resnet18_residual_model_best.pth.tar"
  FILTER_CKPT="${WORK_ROOT}/filter_checkpoints/ucf101_split${split}_residual_fft035_resnet18_residual_checkpoint.pth.tar"

  if [ -f "${FILTER_BEST}" ]; then
    FILTER_WEIGHT="${FILTER_BEST}"
  elif [ -f "${FILTER_CKPT}" ]; then
    FILTER_WEIGHT="${FILTER_CKPT}"
  else
    echo "[ERROR] split${split} 的 filtered residual 权重不存在"
    exit 1
  fi

  python "${WORK_ROOT}/test.py" \
    --data-name ucf101 \
    --data-root "${DATA_ROOT}" \
    --test-list "${TEST_LIST}" \
    --representation residual \
    --arch resnet18 \
    --weights "${FILTER_WEIGHT}" \
    --save-scores "${OUT_DIR}/ucf101_split${split}_residual_fft035.npz" \
    -j 8 \
    --gpus 0 \
    --use_freq_filter \
    --freq_filter_type fft_lowpass \
    --fft_cutoff 0.35 \
    2>&1 | tee "${LOG_DIR}/ucf101_split${split}_residual_fft035_test.log"
done

# -----------------------------
# 3. 原论文固定权重 2:1:1 三流融合
# -----------------------------
for split in 1 2 3; do
  # baseline fusion
  python "${WORK_ROOT}/fixed_fuse_npz.py" \
    --iframe "${OUT_DIR}/ucf101_split${split}_iframe.npz" \
    --mv "${OUT_DIR}/ucf101_split${split}_mv.npz" \
    --residual "${OUT_DIR}/ucf101_split${split}_residual_baseline.npz" \
    --out "${FUSED_DIR}/ucf101_split${split}_fusion_baseline_211.npz" \
    --w_iframe 2 \
    --w_mv 1 \
    --w_residual 1

  # filtered fusion
  python "${WORK_ROOT}/fixed_fuse_npz.py" \
    --iframe "${OUT_DIR}/ucf101_split${split}_iframe.npz" \
    --mv "${OUT_DIR}/ucf101_split${split}_mv.npz" \
    --residual "${OUT_DIR}/ucf101_split${split}_residual_fft035.npz" \
    --out "${FUSED_DIR}/ucf101_split${split}_fusion_fft035_211.npz" \
    --w_iframe 2 \
    --w_mv 1 \
    --w_residual 1
done

# -----------------------------
# 4. 汇总输出
# -----------------------------
python - <<'PY'
import numpy as np
from pathlib import Path

root = Path("/home/srtp/ucf101_filter_work/fusion_scores/fused_211")

def read_acc(path):
    z = np.load(path, allow_pickle=True)
    if 'accuracy' in z.files:
        return float(z['accuracy'])
    return None

base = []
filt = []

print("split,baseline_fusion_211,fft035_fusion_211,delta")
for split in [1, 2, 3]:
    b = read_acc(root / f"ucf101_split{split}_fusion_baseline_211.npz")
    f = read_acc(root / f"ucf101_split{split}_fusion_fft035_211.npz")
    d = None if (b is None or f is None) else (f - b)
    base.append(b)
    filt.append(f)
    print(f"{split},{b:.4f},{f:.4f},{d:.4f}")

mb = sum(base) / len(base)
mf = sum(filt) / len(filt)
print(f"mean,{mb:.4f},{mf:.4f},{(mf-mb):.4f}")
PY

echo
echo "[DONE] 单流结果已整理、.npz 已导出、2:1:1 固定权重三流融合已完成"
echo "单流分数目录: ${OUT_DIR}"
echo "融合分数目录: ${FUSED_DIR}"
echo "日志目录    : ${LOG_DIR}"
