import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.plot_utils import *

os.makedirs("results/plots", exist_ok=True)


# =========================
# LOAD DATA
# =========================
base_preds = np.load("results/baseline_preds.npy")
base_trues = np.load("results/baseline_trues.npy")
base_returns = np.load("results/baseline_returns.npy")

tq_preds = np.load("results/tq_preds.npy")
tq_trues = np.load("results/tq_trues.npy")
tq_returns = np.load("results/tq_returns.npy")
tq_confs = np.load("results/tq_confs.npy")


# =========================
# METRICS
# =========================
def get_metrics(trues, preds, returns):
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    precision = precision_score(trues, preds, average="macro")
    recall = recall_score(trues, preds, average="macro")

    trades = len(returns)

    if trades > 0:
        winrate = sum(1 for r in returns if r > 0) / trades
        avg_trade = np.mean(returns)

        capital = 1.0
        equity = []

        for r in returns:
            capital *= (1 + r)
            equity.append(capital)

        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak

        max_dd = drawdown.min()
        total_return = capital - 1

    else:
        winrate = 0
        avg_trade = 0
        max_dd = 0
        total_return = 0

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,

        "trades": trades,
        "winrate": winrate,
        "avg_trade": avg_trade,
        "return": total_return,
        "dd": max_dd
    }


base_m = get_metrics(base_trues, base_preds, base_returns)
tq_m   = get_metrics(tq_trues, tq_preds, tq_returns)


# =========================
# PLOTS (FINAL SET)
# =========================

# 1. Core result (QUAN TRỌNG NHẤT)
plot_compare_equity_dd(base_returns, tq_returns)

# 2. Trading insight
save_metrics_table(base_m, tq_m)

# 3. ML vs Trading mismatch
plot_compare_classification(base_m, tq_m)

# 4. TQ core idea
plot_confidence_sweep(tq_preds, tq_confs, tq_returns)

# 5. Confusion (TQ only)
plot_confusion(tq_trues, tq_preds, "tq")


# =========================
# PRINT SUMMARY
# =========================
print("\n===== FINAL COMPARISON =====")
for k in base_m:
    print(f"{k:10} | base: {base_m[k]:.4f} | tq: {tq_m[k]:.4f}")

print("\n✅ Plots ready for thesis (clean version)")