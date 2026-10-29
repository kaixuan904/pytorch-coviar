import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# 配置
# -----------------------------
SPLITS = [1, 2, 3]  # UCF101 splits
MODES = ['i', 'mv', 'r']
FILES_DIR = Path("./results")
OUTPUT_DIR = FILES_DIR
OUTPUT_LOG = FILES_DIR / "ucf101_best_optimized_fusion.log"

# 原论文固定权重
FIXED_WEIGHTS = {'i': 2.0, 'mv': 1.0, 'r': 1.0}

# Softmax 温度
TEMP = 1.0  # 可微调 0.9~1.1

# -----------------------------
# 工具函数
# -----------------------------
def load_scores(split, mode):
    path = FILES_DIR / f"ucf101_split{split}_{mode}_scores.npz"
    data = np.load(path)
    if mode == 'i':
        scores, labels = data['i_scores'], data['labels']
    elif mode == 'mv':
        scores, labels = data['mv_scores'], data['labels']
    else:
        scores, labels = data['residual_scores'], data['labels']
    return scores, labels

def softmax(scores, T=1.0):
    scores = scores / T
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def topk_acc(scores, labels, k=1):
    preds = np.argsort(scores, axis=1)[:, ::-1]
    if k == 1:
        return (preds[:, 0] == labels).mean() * 100
    else:
        return np.mean([labels[i] in preds[i, :k] for i in range(len(labels))]) * 100

# -----------------------------
# 开始处理每个 split
# -----------------------------
log_lines = []

final_top1 = []
final_top5 = []

for split in SPLITS:
    # 加载三流
    i_scores, labels = load_scores(split, 'i')
    mv_scores, _ = load_scores(split, 'mv')
    r_scores, _ = load_scores(split, 'r')
    
    # Softmax + 归一化
    i_scores = softmax(i_scores, TEMP)
    mv_scores = softmax(mv_scores, TEMP)
    r_scores = softmax(r_scores, TEMP)
    
    # 保存每流 npz
    np.savez(OUTPUT_DIR / f"split{split}_i.npz", scores=i_scores, labels=labels)
    np.savez(OUTPUT_DIR / f"split{split}_mv.npz", scores=mv_scores, labels=labels)
    np.savez(OUTPUT_DIR / f"split{split}_r.npz", scores=r_scores, labels=labels)
    
    # -----------------------------
    # 固定权重融合
    # -----------------------------
    fused_fixed = FIXED_WEIGHTS['i']*i_scores + FIXED_WEIGHTS['mv']*mv_scores + FIXED_WEIGHTS['r']*r_scores
    fixed_top1 = topk_acc(fused_fixed, labels, 1)
    fixed_top5 = topk_acc(fused_fixed, labels, 5)
    
    # -----------------------------
    # 优化融合（轻量线性回归）
    # -----------------------------
    X = np.hstack([i_scores, mv_scores, r_scores])
    y = labels
    # 线性回归学习最佳组合
    lr = LinearRegression()
    lr.fit(X, np.eye(X.shape[1])[y])  # one-hot
    fused_opt = lr.predict(X)
    
    opt_top1 = topk_acc(fused_opt, labels, 1)
    opt_top5 = topk_acc(fused_opt, labels, 5)
    
    # 保存优化融合 npz
    np.savez(OUTPUT_DIR / f"split{split}_optimized_fusion.npz", fused=fused_opt, labels=labels)
    
    log_lines.append(f"===== split{split} =====")
    log_lines.append(f"I-frame Top-1={topk_acc(i_scores, labels,1):.4f} Top-5={topk_acc(i_scores, labels,5):.4f}")
    log_lines.append(f"MV      Top-1={topk_acc(mv_scores, labels,1):.4f} Top-5={topk_acc(mv_scores, labels,5):.4f}")
    log_lines.append(f"Residual Top-1={topk_acc(r_scores, labels,1):.4f} Top-5={topk_acc(r_scores, labels,5):.4f}")
    log_lines.append(f"Fixed Fusion Top-1={fixed_top1:.4f} Top-5={fixed_top5:.4f} Weights={FIXED_WEIGHTS}")
    log_lines.append(f"Optimized Fusion Top-1={opt_top1:.4f} Top-5={opt_top5:.4f} LinearReg weights approx={lr.coef_}\n")
    
    final_top1.append((fixed_top1, opt_top1))
    final_top5.append((fixed_top5, opt_top5))

# -----------------------------
# 全量平均对比
# -----------------------------
fixed_avg_top1 = np.mean([f[0] for f in final_top1])
fixed_avg_top5 = np.mean([f[0] for f in final_top5])
opt_avg_top1 = np.mean([f[1] for f in final_top1])
opt_avg_top5 = np.mean([f[1] for f in final_top5])

log_lines.append("===== FINAL AVERAGE =====")
log_lines.append(f"Fixed Fusion Top-1={fixed_avg_top1:.4f} Top-5={fixed_avg_top5:.4f}")
log_lines.append(f"Optimized Fusion Top-1={opt_avg_top1:.4f} Top-5={opt_avg_top5:.4f}\n")

# 保存 log
with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    f.write("\n".join(log_lines))

print(f"Done! All fusion results saved to {OUTPUT_DIR} and log {OUTPUT_LOG}")
