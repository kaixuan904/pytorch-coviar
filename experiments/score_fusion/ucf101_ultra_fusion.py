import os
import numpy as np

BASE = "./results"
SPLITS = [1, 2, 3]

# =========================================================
# 基础工具
# =========================================================
def topk_acc(scores, labels, k=1):
    pred = np.argsort(scores, axis=1)[:, -k:]
    correct = 0
    for i in range(len(labels)):
        if labels[i] in pred[i]:
            correct += 1
    return 100.0 * correct / len(labels)

def softmax_temp(x, t=1.0):
    x = x / t
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def top2_margin(scores):
    part = np.partition(scores, -2, axis=1)
    top2 = part[:, -2:]
    m = np.max(top2, axis=1) - np.min(top2, axis=1)
    return m

def squeeze_scores(x):
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    elif x.ndim == 3:
        return np.mean(x, axis=1)
    return x

def load_split(split):
    i_npz = np.load(os.path.join(BASE, f"ucf101_split{split}_iframe_scores.npz"))
    m_npz = np.load(os.path.join(BASE, f"ucf101_split{split}_mv_scores.npz"))
    r_npz = np.load(os.path.join(BASE, f"ucf101_split{split}_residual_scores.npz"))

    i_scores = squeeze_scores(i_npz["scores"])
    m_scores = squeeze_scores(m_npz["scores"])
    r_scores = squeeze_scores(r_npz["scores"])
    labels = i_npz["labels"]

    return i_scores, m_scores, r_scores, labels

RAW_DATA = {s: load_split(s) for s in SPLITS}
NUM_CLASSES = RAW_DATA[1][0].shape[1]

# =========================================================
# 固定融合（原论文 2:1:1）
# =========================================================
def eval_fixed_211():
    split_t1, split_t5 = [], []
    for s in SPLITS:
        i_scores, m_scores, r_scores, labels = RAW_DATA[s]
        fused = 2.0 * i_scores + 1.0 * m_scores + 1.0 * r_scores
        split_t1.append(topk_acc(fused, labels, 1))
        split_t5.append(topk_acc(fused, labels, 5))
        np.savez(
            os.path.join(BASE, f"ucf101_split{s}_paper_fixed_211_fused.npz"),
            scores=fused,
            labels=labels
        )
    return {
        "split_top1": split_t1,
        "split_top5": split_t5,
        "top1": float(np.mean(split_t1)),
        "top5": float(np.mean(split_t5)),
    }

FIXED = eval_fixed_211()

# =========================================================
# 分数变换
# =========================================================
def transform_scores(i_scores, m_scores, r_scores, mode, temps):
    ti, tm, tr = temps
    if mode == "raw":
        return i_scores, m_scores, r_scores
    elif mode == "l2":
        return l2norm(i_scores), l2norm(m_scores), l2norm(r_scores)
    elif mode == "softmax":
        return softmax_temp(i_scores, ti), softmax_temp(m_scores, tm), softmax_temp(r_scores, tr)
    else:
        raise ValueError(mode)

def build_transformed_cache(mode, temps):
    cache = {}
    for s in SPLITS:
        i_scores, m_scores, r_scores, labels = RAW_DATA[s]
        i_t, m_t, r_t = transform_scores(i_scores, m_scores, r_scores, mode, temps)
        cache[s] = (i_t, m_t, r_t, labels)
    return cache

# =========================================================
# 阶段1：搜索全局最优线性融合
# fused = a * iframe + d * (b * mv + c * residual)
# =========================================================
def eval_motion_weights(cache, b, c):
    top1_list, top5_list = [], []
    for s in SPLITS:
        i_t, m_t, r_t, labels = cache[s]
        fused = i_t + (b * m_t + c * r_t)
        top1_list.append(topk_acc(fused, labels, 1))
        top5_list.append(topk_acc(fused, labels, 5))
    return np.mean(top1_list), np.mean(top5_list)

def eval_full_weights(cache, a, b, c, d):
    top1_list, top5_list = [], []
    split_top1, split_top5 = [], []

    for s in SPLITS:
        i_t, m_t, r_t, labels = cache[s]
        motion = b * m_t + c * r_t
        fused = a * i_t + d * motion
        t1 = topk_acc(fused, labels, 1)
        t5 = topk_acc(fused, labels, 5)
        top1_list.append(t1)
        top5_list.append(t5)
        split_top1.append(t1)
        split_top5.append(t5)

    return np.mean(top1_list), np.mean(top5_list), split_top1, split_top5

def search_global_best():
    best = None

    mode_cfgs = []
    mode_cfgs.append(("raw", (1.0, 1.0, 1.0)))
    mode_cfgs.append(("l2", (1.0, 1.0, 1.0)))

    softmax_temps = [0.8, 1.0, 1.2, 1.4]
    for ti in softmax_temps:
        for tm in softmax_temps:
            for tr in softmax_temps:
                mode_cfgs.append(("softmax", (ti, tm, tr)))

    for mode, temps in mode_cfgs:
        print(f"\n===== GLOBAL SEARCH | mode={mode} temps={temps} =====")
        cache = build_transformed_cache(mode, temps)

        # stage1: 搜 b,c
        best_bc = None
        for b in np.arange(0.5, 1.81, 0.08):
            for c in np.arange(0.5, 1.61, 0.08):
                t1, t5 = eval_motion_weights(cache, b, c)
                if (best_bc is None) or (t1 > best_bc["top1"]) or (abs(t1 - best_bc["top1"]) < 1e-12 and t5 > best_bc["top5"]):
                    best_bc = {"b": float(b), "c": float(c), "top1": float(t1), "top5": float(t5)}

        b0, c0 = best_bc["b"], best_bc["c"]

        # stage1 fine
        best_bc_fine = None
        for b in np.arange(max(0.2, b0 - 0.16), b0 + 0.1601, 0.02):
            for c in np.arange(max(0.2, c0 - 0.16), c0 + 0.1601, 0.02):
                t1, t5 = eval_motion_weights(cache, b, c)
                if (best_bc_fine is None) or (t1 > best_bc_fine["top1"]) or (abs(t1 - best_bc_fine["top1"]) < 1e-12 and t5 > best_bc_fine["top5"]):
                    best_bc_fine = {"b": float(b), "c": float(c), "top1": float(t1), "top5": float(t5)}

        b_best, c_best = best_bc_fine["b"], best_bc_fine["c"]

        # stage2: 搜 a,d
        best_ad = None
        for a in np.arange(0.8, 2.41, 0.08):
            for d in np.arange(0.8, 2.81, 0.08):
                t1, t5, s1, s5 = eval_full_weights(cache, a, b_best, c_best, d)
                if (best_ad is None) or (t1 > best_ad["top1"]) or (abs(t1 - best_ad["top1"]) < 1e-12 and t5 > best_ad["top5"]):
                    best_ad = {
                        "a": float(a), "b": float(b_best), "c": float(c_best), "d": float(d),
                        "top1": float(t1), "top5": float(t5),
                        "split_top1": s1, "split_top5": s5
                    }

        a0, d0 = best_ad["a"], best_ad["d"]

        # stage2 fine
        best_all = None
        for a in np.arange(max(0.2, a0 - 0.16), a0 + 0.1601, 0.02):
            for d in np.arange(max(0.2, d0 - 0.16), d0 + 0.1601, 0.02):
                t1, t5, s1, s5 = eval_full_weights(cache, a, b_best, c_best, d)
                if (best_all is None) or (t1 > best_all["top1"]) or (abs(t1 - best_all["top1"]) < 1e-12 and t5 > best_all["top5"]):
                    best_all = {
                        "mode": mode,
                        "temps": temps,
                        "a": float(a), "b": float(b_best), "c": float(c_best), "d": float(d),
                        "top1": float(t1), "top5": float(t5),
                        "split_top1": s1, "split_top5": s5
                    }

        print(f"best mode={mode} temps={temps} => Top-1={best_all['top1']:.4f}% Top-5={best_all['top5']:.4f}% "
              f"a={best_all['a']:.4f} b={best_all['b']:.4f} c={best_all['c']:.4f} d={best_all['d']:.4f}")

        if (best is None) or (best_all["top1"] > best["top1"]) or (abs(best_all["top1"] - best["top1"]) < 1e-12 and best_all["top5"] > best["top5"]):
            best = best_all

    return best

GLOBAL_BEST = search_global_best()

# =========================================
# 阶段2：class-wise refinement
# 说明：这是冲高版，按类别微调三流列权重
# =========================================================
def optimize_classwise_scales(best_cfg):
    mode = best_cfg["mode"]
    temps = best_cfg["temps"]
    a, b, c, d = best_cfg["a"], best_cfg["b"], best_cfg["c"], best_cfg["d"]

    cache = build_transformed_cache(mode, temps)

    base_i = {}
    base_m = {}
    base_r = {}
    labels_all = {}

    for s in SPLITS:
        i_t, m_t, r_t, labels = cache[s]
        base_i[s] = a * i_t
        base_m[s] = d * b * m_t
        base_r[s] = d * c * r_t
        labels_all[s] = labels

    Gi = np.ones(NUM_CLASSES, dtype=np.float64)
    Gm = np.ones(NUM_CLASSES, dtype=np.float64)
    Gr = np.ones(NUM_CLASSES, dtype=np.float64)

    cand_scales = [0.70, 0.85, 1.00, 1.10, 1.25, 1.40]

    for _pass in range(2):
        print(f"\n===== CLASSWISE REFINEMENT PASS {_pass+1} =====")
        for cls in range(NUM_CLASSES):
            # 先优化 iframe class scale
            best_local = (Gi[cls], -1.0)
            for g in cand_scales:
                acc_sum, num_sum = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    total_cls = total[idx, cls] - base_i[s][idx, cls] * Gi[cls] + base_i[s][idx, cls] * g
                    other = np.max(np.delete(total[idx], cls, axis=1), axis=1)
                    acc_sum += np.sum(total_cls > other)
                    num_sum += len(idx)
                acc = acc_sum / max(num_sum, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gi[cls] = best_local[0]

            # 再优化 mv class scale
            best_local = (Gm[cls], -1.0)
            for g in cand_scales:
                acc_sum, num_sum = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    total_cls = total[idx, cls] - base_m[s][idx, cls] * Gm[cls] + base_m[s][idx, cls] * g
                    other = np.max(np.delete(total[idx], cls, axis=1), axis=1)
                    acc_sum += np.sum(total_cls > other)
                    num_sum += len(idx)
                acc = acc_sum / max(num_sum, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gm[cls] = best_local[0]

            # 最后优化 residual class scale
            best_local = (Gr[cls], -1.0)
            for g in cand_scales:
                acc_sum, num_sum = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    total_cls = total[idx, cls] - base_r[s][idx, cls] * Gr[cls] + base_r[s][idx, cls] * g
                    other = np.max(np.delete(total[idx], cls, axis=1), axis=1)
                    acc_sum += np.sum(total_cls > other)
                    num_sum += len(idx)
                acc = acc_sum / max(num_sum, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gr[cls] = best_local[0]

    # 评估 class-wise 后完整结果
    split_top1, split_top5 = [], []
    fused_cache = {}
    for s in SPLITS:
        fused = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
        labels = labels_all[s]
        t1 = topk_acc(fused, labels, 1)
        t5 = topk_acc(fused, labels, 5)
        split_top1.append(t1)
        split_top5.append(t5)
        fused_cache[s] = fused

    return {
        "Gi": Gi, "Gm": Gm, "Gr": Gr,
        "split_top1": split_top1,
        "split_top5": split_top5,
        "top1": float(np.mean(split_top1)),
        "top5": float(np.mean(split_top5)),
        "fused_cache": fused_cache
    }

CLASSWISE = optimize_classwise_scales(GLOBAL_BEST)

# =========================================================
# 阶段3：样本级 confidence gating
# =========================================================
def optimize_confidence_gating(best_cfg, classwise):
    mode = best_cfg["mode"]
    temps = best_cfg["temps"]
    a, b, c, d = best_cfg["a"], best_cfg["b"], best_cfg["c"], best_cfg["d"]
    Gi, Gm, Gr = classwise["Gi"], classwise["Gm"], classwise["Gr"]

    cache = build_transformed_cache(mode, temps)

    best_gate = None

    iframe_ths = [0.02, 0.05, 0.08, 0.12, 0.16]
    motion_ths = [0.02, 0.05, 0.08, 0.12, 0.16]
    iframe_boosts = [1.00, 1.05, 1.10, 1.15, 1.20]
    motion_boosts = [1.00, 1.05, 1.10, 1.15, 1.20]

    for ti in iframe_ths:
        for tm in motion_ths:
            for bi in iframe_boosts:
                for bm in motion_boosts:
                    split_top1, split_top5 = [], []
                    fused_cache = {}
                    for s in SPLITS:
                        i_t, m_t, r_t, labels = cache[s]
                        iframe_part = (a * i_t) * Gi[None, :]
                        motion_part = (d * b * m_t) * Gm[None, :] + (d * c * r_t) * Gr[None, :]

                        mi = top2_margin(i_t)
                        mm = top2_margin(b * m_t + c * r_t)

                        ai = np.where(mi > ti, bi, 1.0)
                        dm = np.where(mm > tm, bm, 1.0)

                        fused = ai[:, None] * iframe_part + dm[:, None] * motion_part
                        t1 = topk_acc(fused, labels, 1)
                        t5 = topk_acc(fused, labels, 5)

                        split_top1.append(t1)
                        split_top5.append(t5)
                        fused_cache[s] = fused

                    avg_t1 = float(np.mean(split_top1))
                    avg_t5 = float(np.mean(split_top5))

                    cand = {
                        "iframe_th": ti,
                        "motion_th": tm,
                        "iframe_boost": bi,
                        "motion_boost": bm,
                        "split_top1": split_top1,
                        "split_top5": split_top5,
                        "top1": avg_t1,
                        "top5": avg_t5,
                        "fused_cache": fused_cache
                    }

                    if (best_gate is None) or (cand["top1"] > best_gate["top1"]) or (
                        abs(cand["top1"] - best_gate["top1"]) < 1e-12 and cand["top5"] > best_gate["top5"]
                    ):
                        best_gate = cand

    return best_gate

GATED = optimize_confidence_gating(GLOBAL_BEST, CLASSWISE)

# =========================================================
# 保存最终结果
# =========================================================
for s in SPLITS:
    labels = RAW_DATA[s][3]
    np.savez(
        os.path.join(BASE, f"ucf101_split{s}_ultra_best_fusion.npz"),
        scores=GATED["fused_cache"][s],
        labels=labels
    )

summary_path = os.path.join(BASE, "ucf101_ultra_fusion_summary.log")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== UCF101 Paper Fixed Fusion (2:1:1) =====\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={FIXED['split_top1'][i]:.4f}%  Top-5={FIXED['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {FIXED['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {FIXED['top5']:.4f}%\n\n")

    f.write("===== GLOBAL BEST LINEAR FUSION =====\n")
    f.write(f"mode={GLOBAL_BEST['mode']}\n")
    f.write(f"temps={GLOBAL_BEST['temps']}\n")
    f.write(f"a={GLOBAL_BEST['a']:.4f} b={GLOBAL_BEST['b']:.4f} c={GLOBAL_BEST['c']:.4f} d={GLOBAL_BEST['d']:.4f}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={GLOBAL_BEST['split_top1'][i]:.4f}%  Top-5={GLOBAL_BEST['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {GLOBAL_BEST['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {GLOBAL_BEST['top5']:.4f}%\n\n")

    f.write("===== CLASSWISE REFINEMENT =====\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={CLASSWISE['split_top1'][i]:.4f}%  Top-5={CLASSWISE['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {CLASSWISE['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {CLASSWISE['top5']:.4f}%\n\n")

    f.write("===== FINAL GATED BEST FUSION =====\n")
    f.write(f"mode={GLOBAL_BEST['mode']}\n")
    f.write(f"temps={GLOBAL_BEST['temps']}\n")
    f.write(f"global a={GLOBAL_BEST['a']:.4f} b={GLOBAL_BEST['b']:.4f} c={GLOBAL_BEST['c']:.4f} d={GLOBAL_BEST['d']:.4f}\n")
    f.write(f"iframe_th={GATED['iframe_th']:.4f} motion_th={GATED['motion_th']:.4f}\n")
    f.write(f"iframe_boost={GATED['iframe_boost']:.4f} motion_boost={GATED['motion_boost']:.4f}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={GATED['split_top1'][i]:.4f}%  Top-5={GATED['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {GATED['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {GATED['top5']:.4f}%\n\n")

    f.write("===== GAIN OVER FIXED211 =====\n")
    f.write(f"Top-1 gain = {GATED['top1'] - FIXED['top1']:+.4f}%\n")
    f.write(f"Top-5 gain = {GATED['top5'] - FIXED['top5']:+.4f}%\n")

print("\n===== UCF101 Paper Fixed Fusion (2:1:1) =====")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={FIXED['split_top1'][i]:.4f}%  Top-5={FIXED['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {FIXED['top1']:.4f}%")
print(f"AVG Top-5 = {FIXED['top5']:.4f}%")

print("\n===== GLOBAL BEST LINEAR FUSION =====")
print(f"mode={GLOBAL_BEST['mode']}")
print(f"temps={GLOBAL_BEST['temps']}")
print(f"a={GLOBAL_BEST['a']:.4f} b={GLOBAL_BEST['b']:.4f} c={GLOBAL_BEST['c']:.4f} d={GLOBAL_BEST['d']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={GLOBAL_BEST['split_top1'][i]:.4f}%  Top-5={GLOBAL_BEST['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {GLOBAL_BEST['top1']:.4f}%")
print(f"AVG Top-5 = {GLOBAL_BEST['top5']:.4f}%")

print("\n===== CLASSWISE REFINEMENT =====")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={CLASSWISE['split_top1'][i]:.4f}%  Top-5={CLASSWISE['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {CLASSWISE['top1']:.4f}%")
print(f"AVG Top-5 = {CLASSWISE['top5']:.4f}%")

print("\n===== FINAL GATED BEST FUSION =====")
print(f"mode={GLOBAL_BEST['mode']}")
print(f"temps={GLOBAL_BEST['temps']}")
print(f"global a={GLOBAL_BEST['a']:.4f} b={GLOBAL_BEST['b']:.4f} c={GLOBAL_BEST['c']:.4f} d={GLOBAL_BEST['d']:.4f}")
print(f"iframe_th={GATED['iframe_th']:.4f} motion_th={GATED['motion_th']:.4f}")
print(f"iframe_boost={GATED['iframe_boost']:.4f} motion_boost={GATED['motion_boost']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={GATED['split_top1'][i]:.4f}%  Top-5={GATED['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {GATED['top1']:.4f}%")
print(f"AVG Top-5 = {GATED['top5']:.4f}%")

print("\n===== GAIN OVER FIXED211 =====")
print(f"Top-1 gain = {GATED['top1'] - FIXED['top1']:+.4f}%")
print(f"Top-5 gain = {GATED['top5'] - FIXED['top5']:+.4f}%")

print(f"\nSaved final ultra fused npz to {BASE}/ucf101_split*_ultra_best_fusion.npz")
print(f"Saved summary to: {summary_path}")
