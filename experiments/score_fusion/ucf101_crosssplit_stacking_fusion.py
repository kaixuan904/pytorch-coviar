import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

FILES_DIR = Path("./results")
SPLITS = [1, 2, 3]

# 你当前固定融合基线
FIXED_TOP1 = 86.9678
FIXED_TOP5 = 97.2584

def load_scores(split, mode):
    path = FILES_DIR / f"ucf101_split{split}_{mode}_scores.npz"
    data = np.load(path)

    scores = data["scores"]
    labels = data["labels"]

    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0, :]
    elif scores.ndim == 3:
        scores = np.mean(scores, axis=1)

    return scores.astype(np.float32), labels.astype(np.int64)

def softmax(scores, T=1.0):
    scores = scores / T
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def l2norm(scores):
    return scores / (np.linalg.norm(scores, axis=1, keepdims=True) + 1e-12)

def topk_acc(scores, labels, k=1):
    pred = np.argsort(scores, axis=1)[:, -k:]
    correct = 0
    for i in range(len(labels)):
        if labels[i] in pred[i]:
            correct += 1
    return 100.0 * correct / len(labels)

def build_features(split, mode="l2", temp=1.0):
    iframe_scores, labels = load_scores(split, "iframe")
    mv_scores, _ = load_scores(split, "mv")
    residual_scores, _ = load_scores(split, "residual")

    if mode == "raw":
        i = iframe_scores
        m = mv_scores
        r = residual_scores
    elif mode == "softmax":
        i = softmax(iframe_scores, temp)
        m = softmax(mv_scores, temp)
        r = softmax(residual_scores, temp)
    elif mode == "l2":
        i = l2norm(iframe_scores)
        m = l2norm(mv_scores)
        r = l2norm(residual_scores)
    elif mode == "softmax_l2":
        i = l2norm(softmax(iframe_scores, temp))
        m = l2norm(softmax(mv_scores, temp))
        r = l2norm(softmax(residual_scores, temp))
    else:
        raise ValueError(mode)

    X = np.concatenate([i, m, r], axis=1)
    return X, labels, i, m, r

def fixed_fusion(i, m, r):
    return 2.0 * i + 1.0 * m + 1.0 * r

def train_and_eval(train_splits, test_split, mode="l2", temp=1.0, model_name="lr"):
    X_train_list, y_train_list = [], []
    for s in train_splits:
        X, y, _, _, _ = build_features(s, mode=mode, temp=temp)
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_test, y_test, i_test, m_test, r_test = build_features(test_split, mode=mode, temp=temp)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    if model_name == "lr":
        model = LogisticRegression(
            C=1.0,
            max_iter=3000,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=None
        )
    elif model_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=42
        )
    else:
        raise ValueError(model_name)

    model.fit(X_train_std, y_train)

    prob = model.predict_proba(X_test_std)
    fixed = fixed_fusion(i_test, m_test, r_test)

    fixed_t1 = topk_acc(fixed, y_test, 1)
    fixed_t5 = topk_acc(fixed, y_test, 5)
    stack_t1 = topk_acc(prob, y_test, 1)
    stack_t5 = topk_acc(prob, y_test, 5)

    return {
        "fixed_t1": fixed_t1,
        "fixed_t5": fixed_t5,
        "stack_t1": stack_t1,
        "stack_t5": stack_t5,
        "labels": y_test,
        "fixed_scores": fixed,
        "stack_scores": prob
    }

def run_all(mode="l2", temp=1.0, model_name="lr", save_tag=""):
    lines = []
    stack_t1_all, stack_t5_all = [], []
    fixed_t1_all, fixed_t5_all = [], []

    for test_split in SPLITS:
        train_splits = [s for s in SPLITS if s != test_split]
        out = train_and_eval(train_splits, test_split, mode=mode, temp=temp, model_name=model_name)

        fixed_t1_all.append(out["fixed_t1"])
        fixed_t5_all.append(out["fixed_t5"])
        stack_t1_all.append(out["stack_t1"])
        stack_t5_all.append(out["stack_t5"])

        np.savez(
            FILES_DIR / f"ucf101_split{test_split}_{save_tag}_stack_fused.npz",
            scores=out["stack_scores"],
            labels=out["labels"]
        )

        lines.append(f"===== split{test_split} =====")
        lines.append(f"Fixed211  Top-1={out['fixed_t1']:.4f}%  Top-5={out['fixed_t5']:.4f}%")
        lines.append(f"Stacking  Top-1={out['stack_t1']:.4f}%  Top-5={out['stack_t5']:.4f}%")
        lines.append("")

    fixed_avg_t1 = np.mean(fixed_t1_all)
    fixed_avg_t5 = np.mean(fixed_t5_all)
    stack_avg_t1 = np.mean(stack_t1_all)
    stack_avg_t5 = np.mean(stack_t5_all)

    lines.append("===== Average over 3 splits =====")
    lines.append(f"Fixed211  AVG Top-1={fixed_avg_t1:.4f}%  AVG Top-5={fixed_avg_t5:.4f}%")
    lines.append(f"Stacking  AVG Top-1={stack_avg_t1:.4f}%  AVG Top-5={stack_avg_t5:.4f}%")
    lines.append("")
    lines.append("===== Gain over Fixed211 =====")
    lines.append(f"Top-1 gain = {stack_avg_t1 - fixed_avg_t1:+.4f}%")
    lines.append(f"Top-5 gain = {stack_avg_t5 - fixed_avg_t5:+.4f}%")

    return {
        "lines": lines,
        "fixed_avg_t1": fixed_avg_t1,
        "fixed_avg_t5": fixed_avg_t5,
        "stack_avg_t1": stack_avg_t1,
        "stack_avg_t5": stack_avg_t5
    }

def main():
    settings = [
        ("raw", 1.0, "lr",  "raw_lr"),
        ("softmax", 1.0, "lr", "softmax_lr"),
        ("softmax", 0.9, "lr", "softmax09_lr"),
        ("softmax", 1.1, "lr", "softmax11_lr"),
        ("l2", 1.0, "lr", "l2_lr"),
        ("softmax_l2", 1.0, "lr", "softmaxl2_lr"),
        ("raw", 1.0, "mlp", "raw_mlp"),
        ("softmax", 1.0, "mlp", "softmax_mlp"),
        ("softmax", 0.9, "mlp", "softmax09_mlp"),
        ("softmax", 1.1, "mlp", "softmax11_mlp"),
        ("l2", 1.0, "mlp", "l2_mlp"),
        ("softmax_l2", 1.0, "mlp", "softmaxl2_mlp"),
    ]

    best = None
    summary_lines = []

    for mode, temp, model_name, save_tag in settings:
        summary_lines.append(f"===== mode={mode} temp={temp} model={model_name} =====")
        out = run_all(mode=mode, temp=temp, model_name=model_name, save_tag=save_tag)
        summary_lines.extend(out["lines"])
        summary_lines.append("")

        if best is None or out["stack_avg_t1"] > best["stack_avg_t1"] or (
            abs(out["stack_avg_t1"] - best["stack_avg_t1"]) < 1e-12 and out["stack_avg_t5"] > best["stack_avg_t5"]
        ):
            best = {
                "mode": mode,
                "temp": temp,
                "model_name": model_name,
                "save_tag": save_tag,
                "stack_avg_t1": out["stack_avg_t1"],
                "stack_avg_t5": out["stack_avg_t5"],
                "fixed_avg_t1": out["fixed_avg_t1"],
                "fixed_avg_t5": out["fixed_avg_t5"],
                "lines": out["lines"]
            }

    summary_lines.append("===== FINAL BEST SETTING =====")
    summary_lines.append(f"mode={best['mode']}")
    summary_lines.append(f"temp={best['temp']}")
    summary_lines.append(f"model={best['model_name']}")
    summary_lines.append(f"save_tag={best['save_tag']}")
    summary_lines.append(f"Fixed211 AVG Top-1={best['fixed_avg_t1']:.4f}%  AVG Top-5={best['fixed_avg_t5']:.4f}%")
    summary_lines.append(f"Stacking AVG Top-1={best['stack_avg_t1']:.4f}%  AVG Top-5={best['stack_avg_t5']:.4f}%")
    summary_lines.append(f"Top-1 gain = {best['stack_avg_t1'] - best['fixed_avg_t1']:+.4f}%")
    summary_lines.append(f"Top-5 gain = {best['stack_avg_t5'] - best['fixed_avg_t5']:+.4f}%")

    out_path = FILES_DIR / "ucf101_crosssplit_stacking_fusion.log"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()

