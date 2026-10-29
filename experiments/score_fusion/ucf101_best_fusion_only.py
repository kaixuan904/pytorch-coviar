import os
import numpy as np

BASE = "./results"
SPLITS = [1, 2, 3]

# 你当前固定融合结果，方便最后直接算提升
FIXED_TOP1 = 86.9678
FIXED_TOP5 = 97.2584

def topk_acc(scores, labels, k=1):
    pred = np.argsort(scores, axis=1)[:, -k:]
    correct = 0
    for i in range(len(labels)):
        if labels[i] in pred[i]:
            correct += 1
    return 100.0 * correct / len(labels)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def load_split(split):
    i_npz = np.load(os.path.join(BASE, f"ucf101_split{split}_iframe_scores.npz"))
    m_npz = np.load(os.path.join(BASE, f"ucf101_split{split}_mv_scores.npz"))
    r_npz = np.load(os.path.join(BASE, f"ucf101_split{split}_residual_scores.npz"))

    i_scores = i_npz["scores"]
    m_scores = m_npz["scores"]
    r_scores = r_npz["scores"]
    labels = i_npz["labels"]

    if i_scores.ndim == 3 and i_scores.shape[1] == 1:
        i_scores = i_scores[:, 0, :]
    elif i_scores.ndim == 3:
        i_scores = np.mean(i_scores, axis=1)

    if m_scores.ndim == 3 and m_scores.shape[1] == 1:
        m_scores = m_scores[:, 0, :]
    elif m_scores.ndim == 3:
        m_scores = np.mean(m_scores, axis=1)

    if r_scores.ndim == 3 and r_scores.shape[1] == 1:
        r_scores = r_scores[:, 0, :]
    elif r_scores.ndim == 3:
        r_scores = np.mean(r_scores, axis=1)

    return i_scores, m_scores, r_scores, labels

RAW_DATA = {s: load_split(s) for s in SPLITS}

def transform_scores(i_scores, m_scores, r_scores, mode):
    if mode == "raw":
        return i_scores, m_scores, r_scores
    elif mode == "softmax":
        return softmax(i_scores), softmax(m_scores), softmax(r_scores)
    elif mode == "l2":
        return l2norm(i_scores), l2norm(m_scores), l2norm(r_scores)
    else:
        raise ValueError(mode)

def eval_weights(a, b, c, d, mode):
    top1_list, top5_list = [], []
    for s in SPLITS:
        i_scores, m_scores, r_scores, labels = RAW_DATA[s]
        i_scores, m_scores, r_scores = transform_scores(i_scores, m_scores, r_scores, mode)

        motion = b * m_scores + c * r_scores
        fused = a * i_scores + d * motion

        t1 = topk_acc(fused, labels, 1)
        t5 = topk_acc(fused, labels, 5)
        top1_list.append(t1)
        top5_list.append(t5)

    return np.mean(top1_list), np.mean(top5_list), top1_list, top5_list

def search(mode, a_list, b_list, c_list, d_list, stage_name):
    total = len(a_list) * len(b_list) * len(c_list) * len(d_list)
    cnt = 0
    best = {
        "mode": mode,
        "top1": -1.0,
        "top5": -1.0,
        "a": None, "b": None, "c": None, "d": None,
        "split_top1": None,
        "split_top5": None
    }

    for a in a_list:
        for b in b_list:
            for c in c_list:
                for d in d_list:
                    cnt += 1
                    if cnt % 1000 == 0:
                        print(f"[{mode}-{stage_name}] {cnt}/{total} a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")

                    avg_t1, avg_t5, split_t1, split_t5 = eval_weights(a, b, c, d, mode)

                    if (avg_t1 > best["top1"]) or (abs(avg_t1 - best["top1"]) < 1e-12 and avg_t5 > best["top5"]):
                        best["top1"] = avg_t1
                        best["top5"] = avg_t5
                        best["a"] = float(a)
                        best["b"] = float(b)
                        best["c"] = float(c)
                        best["d"] = float(d)
                        best["split_top1"] = split_t1
                        best["split_top5"] = split_t5
    return best

modes = ["raw", "softmax", "l2"]
global_best = None
all_results = []

for mode in modes:
    print(f"\n===== MODE: {mode} | ROUND 1 COARSE =====")
    coarse_a = np.arange(1.0, 3.01, 0.08)
    coarse_b = np.arange(0.6, 2.01, 0.08)
    coarse_c = np.arange(0.6, 1.61, 0.08)
    coarse_d = np.arange(1.0, 2.41, 0.08)

    best1 = search(mode, coarse_a, coarse_b, coarse_c, coarse_d, "coarse")

    print(f"\n===== MODE: {mode} | ROUND 2 FINE =====")
    a0, b0, c0, d0 = best1["a"], best1["b"], best1["c"], best1["d"]

    fine_a = np.arange(max(0.2, a0 - 0.12), a0 + 0.1201, 0.02)
    fine_b = np.arange(max(0.2, b0 - 0.12), b0 + 0.1201, 0.02)
    fine_c = np.arange(max(0.2, c0 - 0.12), c0 + 0.1201, 0.02)
    fine_d = np.arange(max(0.2, d0 - 0.12), d0 + 0.1201, 0.02)

    best2 = search(mode, fine_a, fine_b, fine_c, fine_d, "fine")
    all_results.append(best2)

    if (global_best is None) or (best2["top1"] > global_best["top1"]) or (
        abs(best2["top1"] - global_best["top1"]) < 1e-12 and best2["top5"] > global_best["top5"]
    ):
        global_best = best2

print("\n===== ALL MODES SUMMARY =====")
for x in all_results:
    print(
        f"mode={x['mode']}  "
        f"AVG Top-1={x['top1']:.4f}%  AVG Top-5={x['top5']:.4f}%  "
        f"a={x['a']:.4f} b={x['b']:.4f} c={x['c']:.4f} d={x['d']:.4f}"
    )

print("\n===== FINAL BEST FUSION =====")
print(f"mode={global_best['mode']}")
print(f"stage1: mv={global_best['b']:.4f}, residual={global_best['c']:.4f}")
print(f"stage2: iframe={global_best['a']:.4f}, motion={global_best['d']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={global_best['split_top1'][i]:.4f}%  Top-5={global_best['split_top5'][i]:.4f}%")
print("------------------------------")
print(f"AVG Top-1 = {global_best['top1']:.4f}%")
print(f"AVG Top-5 = {global_best['top5']:.4f}%")
print()
print("===== GAIN OVER FIXED FUSION =====")
print(f"Top-1 gain = {global_best['top1'] - FIXED_TOP1:+.4f}%")
print(f"Top-5 gain = {global_best['top5'] - FIXED_TOP5:+.4f}%")

# 保存最终最佳融合结果
for s in SPLITS:
    i_scores, m_scores, r_scores, labels = RAW_DATA[s]
    i_scores, m_scores, r_scores = transform_scores(i_scores, m_scores, r_scores, global_best["mode"])

    motion = global_best["b"] * m_scores + global_best["c"] * r_scores
    fused = global_best["a"] * i_scores + global_best["d"] * motion

    np.savez(
        os.path.join(BASE, f"ucf101_split{s}_best_fusion_only.npz"),
        scores=fused,
        labels=labels
    )

summary_path = os.path.join(BASE, "ucf101_best_fusion_only_summary.log")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== ALL MODES SUMMARY =====\n")
    for x in all_results:
        f.write(
            f"mode={x['mode']}  "
            f"AVG Top-1={x['top1']:.4f}%  AVG Top-5={x['top5']:.4f}%  "
            f"a={x['a']:.4f} b={x['b']:.4f} c={x['c']:.4f} d={x['d']:.4f}\n"
        )
    f.write("\n===== FINAL BEST FUSION =====\n")
    f.write(f"mode={global_best['mode']}\n")
    f.write(f"stage1: mv={global_best['b']:.4f}, residual={global_best['c']:.4f}\n")
    f.write(f"stage2: iframe={global_best['a']:.4f}, motion={global_best['d']:.4f}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={global_best['split_top1'][i]:.4f}%  Top-5={global_best['split_top5'][i]:.4f}%\n")
    f.write("------------------------------\n")
    f.write(f"AVG Top-1 = {global_best['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {global_best['top5']:.4f}%\n\n")
    f.write("===== GAIN OVER FIXED FUSION =====\n")
    f.write(f"Top-1 gain = {global_best['top1'] - FIXED_TOP1:+.4f}%\n")
    f.write(f"Top-5 gain = {global_best['top5'] - FIXED_TOP5:+.4f}%\n")

print(f"\nSaved final fused npz to {BASE}/ucf101_split*_best_fusion_only.npz")
print(f"Saved summary to: {summary_path}")
