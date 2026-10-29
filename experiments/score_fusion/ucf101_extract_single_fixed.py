import os
import numpy as np

SPLITS = [1, 2, 3]

def topk_acc(scores, labels, k=1):
    pred = np.argsort(scores, axis=1)[:, -k:]
    correct = 0
    for i in range(len(labels)):
        if labels[i] in pred[i]:
            correct += 1
    return 100.0 * correct / len(labels)

def load_npz(path):
    x = np.load(path)
    scores = x["scores"]
    labels = x["labels"]

    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0, :]
    elif scores.ndim == 3:
        scores = np.mean(scores, axis=1)

    return scores, labels

i_t1_all, i_t5_all = [], []
m_t1_all, m_t5_all = [], []
r_t1_all, r_t5_all = [], []
f_t1_all, f_t5_all = [], []

lines = []

for s in SPLITS:
    i_scores, labels = load_npz(f"./results/ucf101_split{s}_iframe_scores.npz")
    m_scores, _ = load_npz(f"./results/ucf101_split{s}_mv_scores.npz")
    r_scores, _ = load_npz(f"./results/ucf101_split{s}_residual_scores.npz")

    i_t1 = topk_acc(i_scores, labels, 1)
    i_t5 = topk_acc(i_scores, labels, 5)
    m_t1 = topk_acc(m_scores, labels, 1)
    m_t5 = topk_acc(m_scores, labels, 5)
    r_t1 = topk_acc(r_scores, labels, 1)
    r_t5 = topk_acc(r_scores, labels, 5)

    fused = 2.0 * i_scores + 1.0 * m_scores + 1.0 * r_scores
    f_t1 = topk_acc(fused, labels, 1)
    f_t5 = topk_acc(fused, labels, 5)

    i_t1_all.append(i_t1); i_t5_all.append(i_t5)
    m_t1_all.append(m_t1); m_t5_all.append(m_t5)
    r_t1_all.append(r_t1); r_t5_all.append(r_t5)
    f_t1_all.append(f_t1); f_t5_all.append(f_t5)

    lines.append(f"===== split{s} =====")
    lines.append(f"I-frame   Top-1={i_t1:.4f}%  Top-5={i_t5:.4f}%")
    lines.append(f"MV        Top-1={m_t1:.4f}%  Top-5={m_t5:.4f}%")
    lines.append(f"Residual  Top-1={r_t1:.4f}%  Top-5={r_t5:.4f}%")
    lines.append(f"Fixed211  Top-1={f_t1:.4f}%  Top-5={f_t5:.4f}%")
    lines.append("")

lines.append("===== Average over 3 splits =====")
lines.append(f"I-frame   AVG Top-1={np.mean(i_t1_all):.4f}%  AVG Top-5={np.mean(i_t5_all):.4f}%")
lines.append(f"MV        AVG Top-1={np.mean(m_t1_all):.4f}%  AVG Top-5={np.mean(m_t5_all):.4f}%")
lines.append(f"Residual  AVG Top-1={np.mean(r_t1_all):.4f}%  AVG Top-5={np.mean(r_t5_all):.4f}%")
lines.append(f"Fixed211  AVG Top-1={np.mean(f_t1_all):.4f}%  AVG Top-5={np.mean(f_t5_all):.4f}%")

out = "./results/ucf101_single_fixed_only.log"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved to: {out}")
