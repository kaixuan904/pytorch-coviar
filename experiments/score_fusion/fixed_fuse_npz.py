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
