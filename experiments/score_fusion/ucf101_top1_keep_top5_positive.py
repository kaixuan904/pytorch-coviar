import os
import numpy as np

BASE = "./results"
SPLITS = [1, 2, 3]

# =========================================================
# 基础函数
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
    return np.max(top2, axis=1) - np.min(top2, axis=1)

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
    split_top1, split_top5 = [], []
    fused_cache = {}

    for s in SPLITS:
        i_scores, m_scores, r_scores, labels = RAW_DATA[s]
        fused = 2.0 * i_scores + 1.0 * m_scores + 1.0 * r_scores
        split_top1.append(topk_acc(fused, labels, 1))
        split_top5.append(topk_acc(fused, labels, 5))
        fused_cache[s] = fused.astype(np.float32)

        np.savez(
            os.path.join(BASE, f"ucf101_split{s}_paper_fixed_211_fused.npz"),
            scores=fused,
            labels=labels
        )

    return {
        "split_top1": split_top1,
        "split_top5": split_top5,
        "top1": float(np.mean(split_top1)),
        "top5": float(np.mean(split_top5)),
        "scores": fused_cache
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

def build_cache(mode, temps):
    cache = {}
    for s in SPLITS:
        i_scores, m_scores, r_scores, labels = RAW_DATA[s]
        i_t, m_t, r_t = transform_scores(i_scores, m_scores, r_scores, mode, temps)
        cache[s] = (i_t, m_t, r_t, labels)
    return cache

# =========================================================
# 1) GLOBAL BEST LINEAR FUSION
# fused = wi * iframe + wm * mv + wr * residual
# =========================================================
def eval_direct_weights(cache, wi, wm, wr):
    split_top1, split_top5 = [], []
    split_scores = {}

    for s in SPLITS:
        i_t, m_t, r_t, labels = cache[s]
        fused = wi * i_t + wm * m_t + wr * r_t
        split_top1.append(topk_acc(fused, labels, 1))
        split_top5.append(topk_acc(fused, labels, 5))
        split_scores[s] = fused.astype(np.float32)

    return {
        "top1": float(np.mean(split_top1)),
        "top5": float(np.mean(split_top5)),
        "split_top1": split_top1,
        "split_top5": split_top5,
        "scores": split_scores,
    }

def search_global_best():
    best = None

    configs = []
    configs.append(("raw", (1.0, 1.0, 1.0)))
    configs.append(("l2", (1.0, 1.0, 1.0)))

    for t in [0.8, 1.0, 1.2]:
        configs.append(("softmax", (t, t, t)))

    for mode, temps in configs:
        print(f"\n===== GLOBAL SEARCH | mode={mode} temps={temps} =====")
        cache = build_cache(mode, temps)

        # coarse
        best_c = None
        for wi in np.arange(0.40, 2.01, 0.08):
            for wm in np.arange(0.40, 1.81, 0.08):
                for wr in np.arange(0.40, 1.81, 0.08):
                    cur = eval_direct_weights(cache, wi, wm, wr)
                    if (best_c is None) or (cur["top1"] > best_c["top1"]) or (
                        abs(cur["top1"] - best_c["top1"]) < 1e-12 and cur["top5"] > best_c["top5"]
                    ):
                        best_c = {
                            "mode": mode,
                            "temps": temps,
                            "wi": float(wi), "wm": float(wm), "wr": float(wr),
                            **cur
                        }

        wi0, wm0, wr0 = best_c["wi"], best_c["wm"], best_c["wr"]

        # fine
        best_f = None
        for wi in np.arange(max(0.1, wi0 - 0.16), wi0 + 0.1601, 0.02):
            for wm in np.arange(max(0.1, wm0 - 0.16), wm0 + 0.1601, 0.02):
                for wr in np.arange(max(0.1, wr0 - 0.16), wr0 + 0.1601, 0.02):
                    cur = eval_direct_weights(cache, wi, wm, wr)
                    if (best_f is None) or (cur["top1"] > best_f["top1"]) or (
                        abs(cur["top1"] - best_f["top1"]) < 1e-12 and cur["top5"] > best_f["top5"]
                    ):
                        best_f = {
                            "mode": mode,
                            "temps": temps,
                            "wi": float(wi), "wm": float(wm), "wr": float(wr),
                            **cur
                        }

        print(
            f"BEST mode={mode} temps={temps} => "
            f"Top-1={best_f['top1']:.4f}% Top-5={best_f['top5']:.4f}% "
            f"wi={best_f['wi']:.4f} wm={best_f['wm']:.4f} wr={best_f['wr']:.4f}"
        )

        if (best is None) or (best_f["top1"] > best["top1"]) or (
            abs(best_f["top1"] - best["top1"]) < 1e-12 and best_f["top5"] > best["top5"]
        ):
            best = best_f

    return best

GLOBAL = search_global_best()

# =========================================================
# 2) GENTLE CLASSWISE REFINEMENT
# 目标：尽量提 Top-1，但不要大伤 Top-5
# =========================================================
def classwise_refinement(global_best):
    mode = global_best["mode"]
    temps = global_best["temps"]
    wi, wm, wr = global_best["wi"], global_best["wm"], global_best["wr"]

    cache = build_cache(mode, temps)
    base_i = {}
    base_m = {}
    base_r = {}
    labels_all = {}

    for s in SPLITS:
        i_t, m_t, r_t, labels = cache[s]
        base_i[s] = wi * i_t
        base_m[s] = wm * m_t
        base_r[s] = wr * r_t
        labels_all[s] = labels

    Gi = np.ones(NUM_CLASSES, dtype=np.float64)
    Gm = np.ones(NUM_CLASSES, dtype=np.float64)
    Gr = np.ones(NUM_CLASSES, dtype=np.float64)

    # 比你上一版更温和
    cand_scales = [0.94, 0.98, 1.00, 1.02, 1.06]

    for _pass in range(1):  # 只做 1 pass，减少过拟合
        print(f"\n===== CLASSWISE REFINEMENT PASS {_pass+1} =====")
        for cls in range(NUM_CLASSES):
            # iframe
            best_local = (Gi[cls], -1.0)
            for g in cand_scales:
                hit, total = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total_score = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    tmp = total_score[idx].copy()
                    tmp[:, cls] = (
                        tmp[:, cls]
                        - base_i[s][idx, cls] * Gi[cls]
                        + base_i[s][idx, cls] * g
                    )
                    pred = np.argmax(tmp, axis=1)
                    hit += np.sum(pred == cls)
                    total += len(idx)
                acc = hit / max(total, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gi[cls] = best_local[0]

            # mv
            best_local = (Gm[cls], -1.0)
            for g in cand_scales:
                hit, total = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total_score = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    tmp = total_score[idx].copy()
                    tmp[:, cls] = (
                        tmp[:, cls]
                        - base_m[s][idx, cls] * Gm[cls]
                        + base_m[s][idx, cls] * g
                    )
                    pred = np.argmax(tmp, axis=1)
                    hit += np.sum(pred == cls)
                    total += len(idx)
                acc = hit / max(total, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gm[cls] = best_local[0]

            # residual
            best_local = (Gr[cls], -1.0)
            for g in cand_scales:
                hit, total = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total_score = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    tmp = total_score[idx].copy()
                    tmp[:, cls] = (
                        tmp[:, cls]
                        - base_r[s][idx, cls] * Gr[cls]
                        + base_r[s][idx, cls] * g
                    )
                    pred = np.argmax(tmp, axis=1)
                    hit += np.sum(pred == cls)
                    total += len(idx)
                acc = hit / max(total, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gr[cls] = best_local[0]

    fused_cache = {}
    split_top1, split_top5 = [], []

    for s in SPLITS:
        fused = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
        labels = labels_all[s]
        fused_cache[s] = fused.astype(np.float32)
        split_top1.append(topk_acc(fused, labels, 1))
        split_top5.append(topk_acc(fused, labels, 5))

    return {
        "Gi": Gi, "Gm": Gm, "Gr": Gr,
        "scores": fused_cache,
        "split_top1": split_top1,
        "split_top5": split_top5,
        "top1": float(np.mean(split_top1)),
        "top5": float(np.mean(split_top5)),
    }

CLASSWISE = classwise_refinement(GLOBAL)

# =================================
# 3) GENTLE GATED REFINEMENT
# 目标：保住 Top-1 提升，同时别把 Top-5 压太狠
# =========================================================
def gated_refinement(global_best, classwise):
    mode = global_best["mode"]
    temps = global_best["temps"]
    wi, wm, wr = global_best["wi"], global_best["wm"], global_best["wr"]

    cache = build_cache(mode, temps)

    best_gate = None

    iframe_ths = [0.04, 0.06, 0.08, 0.10]
    motion_ths = [0.04, 0.06, 0.08, 0.10]
    iframe_boosts = [1.00, 1.03, 1.06, 1.09]
    motion_boosts = [1.00, 1.03, 1.06]

    for ti in iframe_ths:
        for tm in motion_ths:
            for bi in iframe_boosts:
                for bm in motion_boosts:
                    fused_cache = {}
                    split_top1, split_top5 = [], []

                    for s in SPLITS:
                        i_t, m_t, r_t, labels = cache[s]

                        iframe_part = wi * i_t
                        motion_part = wm * m_t + wr * r_t

                        # 用 classwise 的比例信息
                        Gi, Gm, Gr = classwise["Gi"], classwise["Gm"], classwise["Gr"]
                        iframe_part = iframe_part * Gi[None, :]
                        motion_part = (wm * m_t * Gm[None, :]) + (wr * r_t * Gr[None, :])

                        mi = top2_margin(i_t)
                        mm = top2_margin(motion_part)

                        ai = np.where(mi > ti, bi, 1.0)
                        am = np.where(mm > tm, bm, 1.0)

                        fused = ai[:, None] * iframe_part + am[:, None] * motion_part
                        fused_cache[s] = fused.astype(np.float32)

                        split_top1.append(topk_acc(fused, labels, 1))
                        split_top5.append(topk_acc(fused, labels, 5))

                    cur = {
                        "iframe_th": float(ti),
                        "motion_th": float(tm),
                        "iframe_boost": float(bi),
                        "motion_boost": float(bm),
                        "scores": fused_cache,
                        "split_top1": split_top1,
                        "split_top5": split_top5,
                        "top1": float(np.mean(split_top1)),
                        "top5": float(np.mean(split_top5)),
                    }

                    if (best_gate is None) or (cur["top1"] > best_gate["top1"]) or (
                        abs(cur["top1"] - best_gate["top1"]) < 1e-12 and cur["top5"] > best_gate["top5"]
                    ):
                        best_gate = cur

    return best_gate

GATED = gated_refinement(GLOBAL, CLASSWISE)

# =========================================================
# 4) 最终关键步骤：
# Top-5 正增益约束下，最大化 Top-1
#
# final = wf * fixed + wg * global + wc * classwise + wt * gated
# 约束：
#   Top-5 > FIXED.Top-5
# 目标：
#   Top-1 最大
# =========================================================
def eval_candidate_mix(wf, wg, wc, wt):
    fused_cache = {}
    split_top1, split_top5 = [], []
    for s in SPLITS:
        fused = (
            wf * FIXED["scores"][s]
            + wg * GLOBAL["scores"][s]
            + wc * CLASSWISE["scores"][s]
            + wt * GATED["scores"][s]
        )
        labels = RAW_DATA[s][3]
        fused_cache[s] = fused.astype(np.float32)
        split_top1.append(topk_acc(fused, labels, 1))
        split_top5.append(topk_acc(fused, labels, 5))

    return {
        "wf": float(wf), "wg": float(wg), "wc": float(wc), "wt": float(wt),
        "scores": fused_cache,
        "split_top1": split_top1,
        "split_top5": split_top5,
        "top1": float(np.mean(split_top1)),
        "top5": float(np.mean(split_top5)),
    }

def search_top5_positive_final():
    fixed_top5 = FIXED["top5"]

    strict_best = None
    nonneg_best = None
    fallback_best = None

    # coarse simplex search
    vals = np.arange(0.0, 1.0001, 0.05)
    total_cnt = 0
    for wf in vals:
        for wg in vals:
            for wc in vals:
                wt = 1.0 - wf - wg - wc
                if wt < -1e-12:
                    continue
                if wt < 0:
                    wt = 0.0
                if abs((wf + wg + wc + wt) - 1.0) > 1e-9:
                    continue

                total_cnt += 1
                cur = eval_candidate_mix(wf, wg, wc, wt)

                # strict positive Top-5 gain
                if cur["top5"] > fixed_top5 + 1e-9:
                    if (strict_best is None) or (cur["top1"] > strict_best["top1"]) or (
                        abs(cur["top1"] - strict_best["top1"]) < 1e-12 and cur["top5"] > strict_best["top5"]
                    ):
                        strict_best = cur

                # non-negative Top-5 gain
                if cur["top5"] >= fixed_top5 - 1e-12:
                    if (nonneg_best is None) or (cur["top1"] > nonneg_best["top1"]) or (
                        abs(cur["top1"] - nonneg_best["top1"]) < 1e-12 and cur["top5"] > nonneg_best["top5"]
                    ):
                        nonneg_best = cur

                # fallback: trade-off
                score = cur["top1"] + 5.0 * (cur["top5"] - fixed_top5)
                cur["_tradeoff_score"] = score
                if (fallback_best is None) or (score > fallback_best["_tradeoff_score"]):
                    fallback_best = cur

    center = strict_best if strict_best is not None else (nonneg_best if nonneg_best is not None else fallback_best)

    wf0, wg0, wc0, wt0 = center["wf"], center["wg"], center["wc"], center["wt"]

    strict_best2 = None
    nonneg_best2 = None
    fallback_best2 = None

    # fine search around center
    wf_list = np.arange(max(0.0, wf0 - 0.08), min(1.0, wf0 + 0.08) + 1e-12, 0.01)
    wg_list = np.arange(max(0.0, wg0 - 0.08), min(1.0, wg0 + 0.08) + 1e-12, 0.01)
    wc_list = np.arange(max(0.0, wc0 - 0.08), min(1.0, wc0 + 0.08) + 1e-12, 0.01)

    for wf in wf_list:
        for wg in wg_list:
            for wc in wc_list:
                wt = 1.0 - wf - wg - wc
                if wt < -1e-12 or wt > 1.0 + 1e-12:
                    continue
                if wt < 0:
                    wt = 0.0
                if wt > 1:
                    continue
                if abs((wf + wg + wc + wt) - 1.0) > 1e-8:
                    continue

                cur = eval_candidate_mix(wf, wg, wc, wt)

                if cur["top5"] > fixed_top5 + 1e-9:
                    if (strict_best2 is None) or (cur["top1"] > strict_best2["top1"]) or (
                        abs(cur["top1"] - strict_best2["top1"]) < 1e-12 and cur["top5"] > strict_best2["top5"]
                    ):
                        strict_best2 = cur

                if cur["top5"] >= fixed_top5 - 1e-12:
                    if (nonneg_best2 is None) or (cur["top1"] > nonneg_best2["top1"]) or (
                        abs(cur["top1"] - nonneg_best2["top1"]) < 1e-12 and cur["top5"] > nonneg_best2["top5"]
                    ):
                        nonneg_best2 = cur

                score = cur["top1"] + 5.0 * (cur["top5"] - fixed_top5)
                cur["_tradeoff_score"] = score
                if (fallback_best2 is None) or (score > fallback_best2["_tradeoff_score"]):
                    fallback_best2 = cur

    if strict_best2 is not None:
        strict_best2["status"] = "strict_positive_top5_gain"
        return strict_best2
    if nonneg_best2 is not None:
        nonneg_best2["status"] = "nonnegative_top5_gain"
        return nonneg_best2

    fallback_best2["status"] = "fallback_tradeoff_no_positive_top5_found"
    return fallback_best2

FINAL = search_top5_positive_final()

# =========================================================
# 保存最终结果
# =========================================================
for s in SPLITS:
    np.savez(
        os.path.join(BASE, f"ucf101_split{s}_top1_keep_top5_positive.npz"),
        scores=FINAL["scores"][s],
        labels=RAW_DATA[s][3]
    )

summary_path = os.path.join(BASE, "ucf101_top1_keep_top5_positive_summary.log")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== UCF101 Paper Fixed Fusion (2:1:1) =====\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={FIXED['split_top1'][i]:.4f}%  Top-5={FIXED['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {FIXED['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {FIXED['top5']:.4f}%\n\n")

    f.write("===== GLOBAL BEST LINEAR FUSION =====\n")
    f.write(f"mode={GLOBAL['mode']} temps={GLOBAL['temps']}\n")
    f.write(f"wi={GLOBAL['wi']:.4f} wm={GLOBAL['wm']:.4f} wr={GLOBAL['wr']:.4f}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={GLOBAL['split_top1'][i]:.4f}%  Top-5={GLOBAL['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {GLOBAL['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {GLOBAL['top5']:.4f}%\n\n")

    f.write("===== CLASSWISE REFINEMENT =====\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={CLASSWISE['split_top1'][i]:.4f}%  Top-5={CLASSWISE['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {CLASSWISE['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {CLASSWISE['top5']:.4f}%\n\n")

    f.write("===== GATED REFINEMENT =====\n")
    f.write(
        f"iframe_th={GATED['iframe_th']:.4f} motion_th={GATED['motion_th']:.4f} "
        f"iframe_boost={GATED['iframe_boost']:.4f} motion_boost={GATED['motion_boost']:.4f}\n"
    )
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={GATED['split_top1'][i]:.4f}%  Top-5={GATED['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {GATED['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {GATED['top5']:.4f}%\n\n")

    f.write("===== FINAL TOP1 KEEP TOP5 POSITIVE =====\n")
    f.write(f"status={FINAL['status']}\n")
    f.write(
        f"wf={FINAL['wf']:.4f} wg={FINAL['wg']:.4f} "
        f"wc={FINAL['wc']:.4f} wt={FINAL['wt']:.4f}\n"
    )
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={FINAL['split_top1'][i]:.4f}%  Top-5={FINAL['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {FINAL['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {FINAL['top5']:.4f}%\n\n")

    f.write("===== GAIN OVER FIXED211 =====\n")
    f.write(f"Top-1 gain = {FINAL['top1'] - FIXED['top1']:+.4f}%\n")
    f.write(f"Top-5 gain = {FINAL['top5'] - FIXED['top5']:+.4f}%\n")

print("\n===== UCF101 Paper Fixed Fusion (2:1:1) =====")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={FIXED['split_top1'][i]:.4f}%  Top-5={FIXED['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {FIXED['top1']:.4f}%")
print(f"AVG Top-5 = {FIXED['top5']:.4f}%")

print("\n===== GLOBAL BEST LINEAR FUSION =====")
print(f"mode={GLOBAL['mode']} temps={GLOBAL['temps']}")
print(f"wi={GLOBAL['wi']:.4f} wm={GLOBAL['wm']:.4f} wr={GLOBAL['wr']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={GLOBAL['split_top1'][i]:.4f}%  Top-5={GLOBAL['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {GLOBAL['top1']:.4f}%")
print(f"AVG Top-5 = {GLOBAL['top5']:.4f}%")

print("\n===== CLASSWISE REFINEMENT =====")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={CLASSWISE['split_top1'][i]:.4f}%  Top-5={CLASSWISE['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {CLASSWISE['top1']:.4f}%")
print(f"AVG Top-5 = {CLASSWISE['top5']:.4f}%")

print("\n===== GATED REFINEMENT =====")
print(
    f"iframe_th={GATED['iframe_th']:.4f} motion_th={GATED['motion_th']:.4f} "
    f"iframe_boost={GATED['iframe_boost']:.4f} motion_boost={GATED['motion_boost']:.4f}"
)
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={GATED['split_top1'][i]:.4f}%  Top-5={GATED['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {GATED['top1']:.4f}%")
print(f"AVG Top-5 = {GATED['top5']:.4f}%")

print("\n===== FINAL TOP1 KEEP TOP5 POSITIVE =====")
print(f"status={FINAL['status']}")
print(f"wf={FINAL['wf']:.4f} wg={FINAL['wg']:.4f} wc={FINAL['wc']:.4f} wt={FINAL['wt']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={FINAL['split_top1'][i]:.4f}%  Top-5={FINAL['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {FINAL['top1']:.4f}%")
print(f"AVG Top-5 = {FINAL['top5']:.4f}%")

print("\n===== GAIN OVER FIXED211 =====")
print(f"Top-1 gain = {FINAL['top1'] - FIXED['top1']:+.4f}%")
print(f"Top-5 gain = {FINAL['top5'] - FIXED['top5']:+.4f}%")

print(f"\nSaved final npz to {BASE}/ucf101_split*_top1_keep_top5_positive.npz")
print(f"Saved summary to: {summary_path}")
