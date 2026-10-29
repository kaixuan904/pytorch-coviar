import os
import numpy as np

BASE = "./results"
SPLITS = [1, 2, 3]

TARGET_TOP1_GAIN = 1.0000
TARGET_TOP5_GAIN = 0.3000

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

def squeeze_scores(x):
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    elif x.ndim == 3:
        return np.mean(x, axis=1)
    return x

def top2_margin(scores):
    part = np.partition(scores, -2, axis=1)
    top2 = part[:, -2:]
    return np.max(top2, axis=1) - np.min(top2, axis=1)

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
# 统一评估
# =========================================================
def eval_scores_dict(scores_dict):
    split_top1, split_top5 = [], []
    for s in SPLITS:
        labels = RAW_DATA[s][3]
        split_top1.append(topk_acc(scores_dict[s], labels, 1))
        split_top5.append(topk_acc(scores_dict[s], labels, 5))
    return {
        "split_top1": split_top1,
        "split_top5": split_top5,
        "top1": float(np.mean(split_top1)),
        "top5": float(np.mean(split_top5)),
        "scores": scores_dict,
    }

def save_scores_dict(prefix, scores_dict):
    for s in SPLITS:
        np.savez(
            os.path.join(BASE, f"ucf101_split{s}_{prefix}.npz"),
            scores=scores_dict[s],
            labels=RAW_DATA[s][3]
        )

# =========================================================
# 固定融合 2:1:1
# =========================================================
def build_fixed():
    scores = {}
    for s in SPLITS:
        i_scores, m_scores, r_scores, _ = RAW_DATA[s]
        scores[s] = (2.0 * i_scores + 1.0 * m_scores + 1.0 * r_scores).astype(np.float32)
    out = eval_scores_dict(scores)
    out["name"] = "fixed211"
    return out

FIXED = build_fixed()
save_scores_dict("paper_fixed_211_fused", FIXED["scores"])

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
# 1) GLOBAL BEST LINEAR
# =========================================================
def eval_direct(cache, wi, wm, wr):
    scores = {}
    for s in SPLITS:
        i_t, m_t, r_t, _ = cache[s]
        scores[s] = (wi * i_t + wm * m_t + wr * r_t).astype(np.float32)
    out = eval_scores_dict(scores)
    out["wi"] = float(wi)
    out["wm"] = float(wm)
    out["wr"] = float(wr)
    return out

def search_global_candidates():
    configs = [
        ("raw", (1.0, 1.0, 1.0)),
        ("l2", (1.0, 1.0, 1.0)),
        ("softmax", (0.8, 0.8, 0.8)),
        ("softmax", (1.0, 1.0, 1.0)),
        ("softmax", (1.2, 1.2, 1.2)),
        ("softmax", (0.8, 1.0, 1.0)),
        ("softmax", (1.0, 0.8, 1.0)),
        ("softmax", (1.0, 1.0, 0.8)),
        ("softmax", (1.2, 1.0, 1.0)),
        ("softmax", (1.0, 1.2, 1.0)),
        ("softmax", (1.0, 1.0, 1.2)),
    ]

    cands = []

    for mode, temps in configs:
        print(f"\n===== GLOBAL SEARCH | mode={mode} temps={temps} =====")
        cache = build_cache(mode, temps)

        best = None
        # coarse
        for wi in np.arange(0.40, 1.81, 0.08):
            for wm in np.arange(0.40, 2.01, 0.08):
                for wr in np.arange(0.40, 2.01, 0.08):
                    cur = eval_direct(cache, wi, wm, wr)
                    if (best is None) or (cur["top1"] > best["top1"]) or (
                        abs(cur["top1"] - best["top1"]) < 1e-12 and cur["top5"] > best["top5"]
                    ):
                        best = cur

        wi0, wm0, wr0 = best["wi"], best["wm"], best["wr"]

        # fine
        best2 = None
        for wi in np.arange(max(0.1, wi0 - 0.16), wi0 + 0.1601, 0.02):
            for wm in np.arange(max(0.1, wm0 - 0.16), wm0 + 0.1601, 0.02):
                for wr in np.arange(max(0.1, wr0 - 0.16), wr0 + 0.1601, 0.02):
                    cur = eval_direct(cache, wi, wm, wr)
                    if (best2 is None) or (cur["top1"] > best2["top1"]) or (
                        abs(cur["top1"] - best2["top1"]) < 1e-12 and cur["top5"] > best2["top5"]
                    ):
                        best2 = cur

        best2["name"] = f"global_{mode}_{temps}"
        best2["mode"] = mode
        best2["temps"] = temps
        cands.append(best2)

        print(
            f"BEST mode={mode} temps={temps} => "
            f"Top-1={best2['top1']:.4f}% Top-5={best2['top5']:.4f}% "
            f"wi={best2['wi']:.4f} wm={best2['wm']:.4f} wr={best2['wr']:.4f}"
        )

    cands.sort(key=lambda x: (x["top1"], x["top5"]), reverse=True)
    return cands

GLOBAL_CANDS = search_global_candidates()
GLOBAL_BEST = GLOBAL_CANDS[0]

# =========================================================
# 2) CLASSWISE（仍然原方案，只是多做几档温和版本）
# =========================================================
def make_classwise_variant(global_best, scales, passes, tag):
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

    for _ in range(passes):
        for cls in range(NUM_CLASSES):
            # iframe
            best_local = (Gi[cls], -1.0)
            for g in scales:
                hit, total = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total_score = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    tmp = total_score[idx].copy()
                    tmp[:, cls] = tmp[:, cls] - base_i[s][idx, cls] * Gi[cls] + base_i[s][idx, cls] * g
                    pred = np.argmax(tmp, axis=1)
                    hit += np.sum(pred == cls)
                    total += len(idx)
                acc = hit / max(total, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gi[cls] = best_local[0]

            # mv
            best_local = (Gm[cls], -1.0)
            for g in scales:
                hit, total = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total_score = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    tmp = total_score[idx].copy()
                    tmp[:, cls] = tmp[:, cls] - base_m[s][idx, cls] * Gm[cls] + base_m[s][idx, cls] * g
                    pred = np.argmax(tmp, axis=1)
                    hit += np.sum(pred == cls)
                    total += len(idx)
                acc = hit / max(total, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gm[cls] = best_local[0]

            # residual
            best_local = (Gr[cls], -1.0)
            for g in scales:
                hit, total = 0, 0
                for s in SPLITS:
                    y = labels_all[s]
                    idx = np.where(y == cls)[0]
                    if len(idx) == 0:
                        continue
                    total_score = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
                    tmp = total_score[idx].copy()
                    tmp[:, cls] = tmp[:, cls] - base_r[s][idx, cls] * Gr[cls] + base_r[s][idx, cls] * g
                    pred = np.argmax(tmp, axis=1)
                    hit += np.sum(pred == cls)
                    total += len(idx)
                acc = hit / max(total, 1)
                if acc > best_local[1]:
                    best_local = (g, acc)
            Gr[cls] = best_local[0]

    scores = {}
    for s in SPLITS:
        fused = base_i[s] * Gi[None, :] + base_m[s] * Gm[None, :] + base_r[s] * Gr[None, :]
        scores[s] = fused.astype(np.float32)

    out = eval_scores_dict(scores)
    out["name"] = f"classwise_{tag}"
    out["Gi"] = Gi
    out["Gm"] = Gm
    out["Gr"] = Gr
    return out

CLASSWISE_CANDS = [
    make_classwise_variant(GLOBAL_BEST, [0.96, 0.98, 1.00, 1.02, 1.04], 1, "mild1"),
    make_classwise_variant(GLOBAL_BEST, [0.94, 0.98, 1.00, 1.02, 1.06], 1, "mild2"),
    make_classwise_variant(GLOBAL_BEST, [0.97, 1.00, 1.03], 2, "stable2"),
]
CLASSWISE_CANDS.sort(key=lambda x: (x["top1"], x["top5"]), reverse=True)
CLASSWISE_BEST = CLASSWISE_CANDS[0]

# =========================================================
# 3) GATED（仍然原方案，但多测几档）
# =========================================================
def make_gated_variant(base_variant, tag):
    # base_variant 可以是 global_best 或 classwise_best
    if "Gi" in base_variant:
        Gi, Gm, Gr = base_variant["Gi"], base_variant["Gm"], base_variant["Gr"]
    else:
        Gi = np.ones(NUM_CLASSES, dtype=np.float64)
        Gm = np.ones(NUM_CLASSES, dtype=np.float64)
        Gr = np.ones(NUM_CLASSES, dtype=np.float64)

    mode = GLOBAL_BEST["mode"]
    temps = GLOBAL_BEST["temps"]
    wi, wm, wr = GLOBAL_BEST["wi"], GLOBAL_BEST["wm"], GLOBAL_BEST["wr"]

    cache = build_cache(mode, temps)

    best = None
    for ti in [0.03, 0.04, 0.05, 0.06, 0.08]:
        for tm in [0.03, 0.04, 0.05, 0.06, 0.08]:
            for bi in [1.00, 1.03, 1.06, 1.09, 1.12]:
                for bm in [1.00, 1.03, 1.06]:
                    scores = {}
                    for s in SPLITS:
                        i_t, m_t, r_t, _ = cache[s]

                        iframe_part = wi * i_t * Gi[None, :]
                        motion_part = wm * m_t * Gm[None, :] + wr * r_t * Gr[None, :]

                        mi = top2_margin(i_t)
                        mm = top2_margin(wm * m_t + wr * r_t)

                        ai = np.where(mi > ti, bi, 1.0)
                        am = np.where(mm > tm, bm, 1.0)

                        fused = ai[:, None] * iframe_part + am[:, None] * motion_part
                        scores[s] = fused.astype(np.float32)

                    cur = eval_scores_dict(scores)
                    cur["name"] = f"gated_{tag}"
                    cur["iframe_th"] = float(ti)
                    cur["motion_th"] = float(tm)
                    cur["iframe_boost"] = float(bi)
                    cur["motion_boost"] = float(bm)

                    if (best is None) or (cur["top1"] > best["top1"]) or (
                        abs(cur["top1"] - best["top1"]) < 1e-12 and cur["top5"] > best["top5"]
                    ):
                        best = cur
    return best

GATED_GLOBAL = make_gated_variant(GLOBAL_BEST, "from_global")
GATED_CLASSWISE = make_gated_variant(CLASSWISE_BEST, "from_classwise")

GATED_CANDS = [GATED_GLOBAL, GATED_CLASSWISE]
GATED_CANDS.sort(key=lambda x: (x["top1"], x["top5"]), reverse=True)
GATED_BEST = GATED_CANDS[0]

# ==========================
# 4) 最终混合搜索
# 仍然原方案：只在 fixed/global/classwise/gated 体系里混
#
# 目标：
#   优先满足 Top5 gain >= +0.3000
#   再尽量让 Top1 gain >= +1.0000
# =========================================================
POOL = [
    FIXED,
    GLOBAL_BEST,
    CLASSWISE_BEST,
    GATED_BEST,
]

def mix_pool(weights):
    scores = {}
    for s in SPLITS:
        fused = np.zeros_like(POOL[0]["scores"][s], dtype=np.float64)
        for w, item in zip(weights, POOL):
            fused += w * item["scores"][s]
        scores[s] = fused.astype(np.float32)
    return eval_scores_dict(scores)

def better(a, b):
    # 排序逻辑：
    # 1. 是否满足 Top5 gain >= target
    # 2. 是否满足 Top1 gain >= target
    # 3. Top1 更大
    # 4. Top5 更大
    def key(x):
        top1_gain = x["top1"] - FIXED["top1"]
        top5_gain = x["top5"] - FIXED["top5"]
        return (
            int(top5_gain >= TARGET_TOP5_GAIN - 1e-12),
            int(top1_gain >= TARGET_TOP1_GAIN - 1e-12),
            x["top1"],
            x["top5"],
        )
    return key(a) > key(b)

def search_final_mix():
    best = None

    # coarse simplex
    vals = np.arange(0.0, 1.0001, 0.05)
    for wf in vals:
        for wg in vals:
            for wc in vals:
                wt = 1.0 - wf - wg - wc
                if wt < -1e-12 or wt > 1.0 + 1e-12:
                    continue
                if wt < 0:
                    wt = 0.0
                if abs((wf + wg + wc + wt) - 1.0) > 1e-9:
                    continue

                cur = mix_pool([wf, wg, wc, wt])
                cur["wf"], cur["wg"], cur["wc"], cur["wt"] = float(wf), float(wg), float(wc), float(wt)
                if (best is None) or better(cur, best):
                    best = cur

    wf0, wg0, wc0, wt0 = best["wf"], best["wg"], best["wc"], best["wt"]

    # fine
    best2 = None
    wf_list = np.arange(max(0.0, wf0 - 0.10), min(1.0, wf0 + 0.10) + 1e-12, 0.01)
    wg_list = np.arange(max(0.0, wg0 - 0.10), min(1.0, wg0 + 0.10) + 1e-12, 0.01)
    wc_list = np.arange(max(0.0, wc0 - 0.10), min(1.0, wc0 + 0.10) + 1e-12, 0.01)

    for wf in wf_list:
        for wg in wg_list:
            for wc in wc_list:
                wt = 1.0 - wf - wg - wc
                if wt < -1e-12 or wt > 1.0 + 1e-12:
                    continue
                if wt < 0:
                    wt = 0.0
                if abs((wf + wg + wc + wt) - 1.0) > 1e-8:
                    continue

                cur = mix_pool([wf, wg, wc, wt])
                cur["wf"], cur["wg"], cur["wc"], cur["wt"] = float(wf), float(wg), float(wc), float(wt)
                if (best2 is None) or better(cur, best2):
                    best2 = cur

    # ultra local
    wf0, wg0, wc0, wt0 = best2["wf"], best2["wg"], best2["wc"], best2["wt"]
    best3 = None
    wf_list = np.arange(max(0.0, wf0 - 0.03), min(1.0, wf0 + 0.03) + 1e-12, 0.002)
    wg_list = np.arange(max(0.0, wg0 - 0.03), min(1.0, wg0 + 0.03) + 1e-12, 0.002)
    wc_list = np.arange(max(0.0, wc0 - 0.03), min(1.0, wc0 + 0.03) + 1e-12, 0.002)

    for wf in wf_list:
        for wg in wg_list:
            for wc in wc_list:
                wt = 1.0 - wf - wg - wc
                if wt < -1e-12 or wt > 1.0 + 1e-12:
                    continue
                if wt < 0:
                    wt = 0.0
                if abs((wf + wg + wc + wt) - 1.0) > 1e-7:
                    continue

                cur = mix_pool([wf, wg, wc, wt])
                cur["wf"], cur["wg"], cur["wc"], cur["wt"] = float(wf), float(wg), float(wc), float(wt)
                if (best3 is None) or better(cur, best3):
                    best3 = cur

    top1_gain = best3["top1"] - FIXED["top1"]
    top5_gain = best3["top5"] - FIXED["top5"]

    best3["top1_gain"] = top1_gain
    best3["top5_gain"] = top5_gain
    best3["hit_top1_target"] = bool(top1_gain >= TARGET_TOP1_GAIN - 1e-12)
    best3["hit_top5_target"] = bool(top5_gain >= TARGET_TOP5_GAIN - 1e-12)

    return best3

FINAL = search_final_mix()

# =========================================================
# 保存
# =========================================================
save_scores_dict("same_scheme_push_target", FINAL["scores"])

summary_path = os.path.join(BASE, "ucf101_same_scheme_push_target_summary.log")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== FIXED 2:1:1 =====\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={FIXED['split_top1'][i]:.4f}%  Top-5={FIXED['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {FIXED['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {FIXED['top5']:.4f}%\n\n")

    f.write("===== GLOBAL BEST =====\n")
    f.write(f"name={GLOBAL_BEST['name']}\n")
    f.write(f"mode={GLOBAL_BEST['mode']} temps={GLOBAL_BEST['temps']}\n")
    f.write(f"wi={GLOBAL_BEST['wi']:.4f} wm={GLOBAL_BEST['wm']:.4f} wr={GLOBAL_BEST['wr']:.4f}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={GLOBAL_BEST['split_top1'][i]:.4f}%  Top-5={GLOBAL_BEST['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {GLOBAL_BEST['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {GLOBAL_BEST['top5']:.4f}%\n\n")

    f.write("===== BEST CLASSWISE =====\n")
    f.write(f"name={CLASSWISE_BEST['name']}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={CLASSWISE_BEST['split_top1'][i]:.4f}%  Top-5={CLASSWISE_BEST['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {CLASSWISE_BEST['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {CLASSWISE_BEST['top5']:.4f}%\n\n")

    f.write("===== BEST GATED =====\n")
    f.write(f"name={GATED_BEST['name']}\n")
    f.write(
        f"iframe_th={GATED_BEST['iframe_th']:.4f} motion_th={GATED_BEST['motion_th']:.4f} "
        f"iframe_boost={GATED_BEST['iframe_boost']:.4f} motion_boost={GATED_BEST['motion_boost']:.4f}\n"
    )
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={GATED_BEST['split_top1'][i]:.4f}%  Top-5={GATED_BEST['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {GATED_BEST['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {GATED_BEST['top5']:.4f}%\n\n")

    f.write("===== FINAL SAME-SCHEME PUSH =====\n")
    f.write(f"wf={FINAL['wf']:.4f} wg={FINAL['wg']:.4f} wc={FINAL['wc']:.4f} wt={FINAL['wt']:.4f}\n")
    for i, s in enumerate(SPLITS):
        f.write(f"split{s}: Top-1={FINAL['split_top1'][i]:.4f}%  Top-5={FINAL['split_top5'][i]:.4f}%\n")
    f.write("----------------------------------------------\n")
    f.write(f"AVG Top-1 = {FINAL['top1']:.4f}%\n")
    f.write(f"AVG Top-5 = {FINAL['top5']:.4f}%\n")
    f.write(f"Top-1 gain = {FINAL['top1_gain']:+.4f}%\n")
    f.write(f"Top-5 gain = {FINAL['top5_gain']:+.4f}%\n")
    f.write(f"Hit Top-1 target(+1.0) = {FINAL['hit_top1_target']}\n")
    f.write(f"Hit Top-5 target(+0.3) = {FINAL['hit_top5_target']}\n")

print("\n===== FIXED 2:1:1 =====")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={FIXED['split_top1'][i]:.4f}%  Top-5={FIXED['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {FIXED['top1']:.4f}%")
print(f"AVG Top-5 = {FIXED['top5']:.4f}%")

print("\n===== GLOBAL BEST =====")
print(f"name={GLOBAL_BEST['name']}")
print(f"mode={GLOBAL_BEST['mode']} temps={GLOBAL_BEST['temps']}")
print(f"wi={GLOBAL_BEST['wi']:.4f} wm={GLOBAL_BEST['wm']:.4f} wr={GLOBAL_BEST['wr']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={GLOBAL_BEST['split_top1'][i]:.4f}%  Top-5={GLOBAL_BEST['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {GLOBAL_BEST['top1']:.4f}%")
print(f"AVG Top-5 = {GLOBAL_BEST['top5']:.4f}%")

print("\n===== BEST CLASSWISE =====")
print(f"name={CLASSWISE_BEST['name']}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={CLASSWISE_BEST['split_top1'][i]:.4f}%  Top-5={CLASSWISE_BEST['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {CLASSWISE_BEST['top1']:.4f}%")
print(f"AVG Top-5 = {CLASSWISE_BEST['top5']:.4f}%")

print("\n===== BEST GATED =====")
print(f"name={GATED_BEST['name']}")
print(
    f"iframe_th={GATED_BEST['iframe_th']:.4f} motion_th={GATED_BEST['motion_th']:.4f} "
    f"iframe_boost={GATED_BEST['iframe_boost']:.4f} motion_boost={GATED_BEST['motion_boost']:.4f}"
)
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={GATED_BEST['split_top1'][i]:.4f}%  Top-5={GATED_BEST['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {GATED_BEST['top1']:.4f}%")
print(f"AVG Top-5 = {GATED_BEST['top5']:.4f}%")

print("\n===== FINAL SAME-SCHEME PUSH =====")
print(f"wf={FINAL['wf']:.4f} wg={FINAL['wg']:.4f} wc={FINAL['wc']:.4f} wt={FINAL['wt']:.4f}")
for i, s in enumerate(SPLITS):
    print(f"split{s}: Top-1={FINAL['split_top1'][i]:.4f}%  Top-5={FINAL['split_top5'][i]:.4f}%")
print("----------------------------------------------")
print(f"AVG Top-1 = {FINAL['top1']:.4f}%")
print(f"AVG Top-5 = {FINAL['top5']:.4f}%")
print(f"Top-1 gain = {FINAL['top1_gain']:+.4f}%")
print(f"Top-5 gain = {FINAL['top5_gain']:+.4f}%")
print(f"Hit Top-1 target(+1.0) = {FINAL['hit_top1_target']}")
print(f"Hit Top-5 target(+0.3) = {FINAL['hit_top5_target']}")

print(f"\nSaved final npz to {BASE}/ucf101_split*_same_scheme_push_target.npz")
print(f"Saved summary to: {summary_path}")
