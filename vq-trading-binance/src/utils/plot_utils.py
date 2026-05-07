import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =========================
# CONFUSION MATRIX (TQ ONLY dùng chính)
# =========================
def plot_confusion(y_true, y_pred, tag="tq"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{tag.upper()} - Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"results/plots/{tag}_confusion.png")
    plt.close()


# =========================
# BUILD EQUITY + DRAWDOWN
# =========================
def build_equity(returns):
    capital = 1.0
    equity = []

    for r in returns:
        capital *= (1 + r)
        equity.append(capital)

    equity = np.array(equity)

    if len(equity) == 0:
        return np.array([]), np.array([])

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    return equity, drawdown


# =========================
# COMPARE EQUITY + DRAWDOWN (QUAN TRỌNG NHẤT)
# =========================
def plot_compare_equity_dd(base_returns, tq_returns):
    base_eq, base_dd = build_equity(base_returns)
    tq_eq, tq_dd = build_equity(tq_returns)

    fig, axes = plt.subplots(2, 1, figsize=(8,6))

    # Equity
    axes[0].plot(base_eq, label="Baseline")
    axes[0].plot(tq_eq, label="TurboQuant")
    axes[0].set_title("Equity Curve Comparison")
    axes[0].legend()

    # Drawdown
    axes[1].plot(base_dd, label="Baseline")
    axes[1].plot(tq_dd, label="TurboQuant")
    axes[1].set_title("Drawdown Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/plots/compare_equity_dd.png")
    plt.close()


# =========================
# COMPARE CLASSIFICATION
# =========================
def plot_compare_classification(base_m, tq_m):
    labels = ["Accuracy", "F1", "Precision", "Recall"]

    base_vals = [
        base_m["acc"], base_m["f1"],
        base_m["precision"], base_m["recall"]
    ]

    tq_vals = [
        tq_m["acc"], tq_m["f1"],
        tq_m["precision"], tq_m["recall"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, base_vals, width, label="Baseline")
    plt.bar(x + width/2, tq_vals, width, label="TurboQuant")

    plt.xticks(x, labels)
    plt.title("Classification Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/plots/compare_classification.png")
    plt.close()


# =========================
# SAVE METRICS TABLE PNG
# =========================
def save_metrics_table(base_m, tq_m):

    df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "F1 Macro",
            "Precision",
            "Recall",
            "Trades",
            "Winrate",
            "Avg Trade",
            "Total Return",
            "Max Drawdown"
        ],

        "Baseline": [
            f"{base_m['acc']:.4f}",
            f"{base_m['f1']:.4f}",
            f"{base_m['precision']:.4f}",
            f"{base_m['recall']:.4f}",
            f"{base_m['trades']}",
            f"{base_m['winrate']:.4f}",
            f"{base_m['avg_trade']:.6f}",
            f"{base_m['return']:.4f}",
            f"{base_m['dd']:.4f}"
        ],

        "TurboQuant": [
            f"{tq_m['acc']:.4f}",
            f"{tq_m['f1']:.4f}",
            f"{tq_m['precision']:.4f}",
            f"{tq_m['recall']:.4f}",
            f"{tq_m['trades']}",
            f"{tq_m['winrate']:.4f}",
            f"{tq_m['avg_trade']:.6f}",
            f"{tq_m['return']:.4f}",
            f"{tq_m['dd']:.4f}"
        ]
    })

    # ===== DRAW TABLE =====
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    table.scale(1, 1.5)

    plt.title("Final Metrics Comparison", pad=20)

    plt.tight_layout()

    plt.savefig(
        "results/plots/final_metrics_table.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n✅ Saved metrics table PNG")

# =========================
# CONFIDENCE SWEEP (KEY IDEA)
# =========================
def plot_confidence_sweep(preds, confs, returns,
                         thresholds=np.arange(0.3, 0.8, 0.05)):

    trades_list = []
    winrate_list = []
    return_list = []

    for th in thresholds:
        trades = 0
        wins = 0
        capital = 1.0

        for p, c, r in zip(preds, confs, returns):
            if c < th:
                continue

            trades += 1

            if r > 0:
                wins += 1

            capital *= (1 + r)

        trades_list.append(trades)
        winrate_list.append(wins / trades if trades else 0)
        return_list.append(capital - 1)

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, trades_list, label="Trades")
    plt.plot(thresholds, winrate_list, label="Winrate")
    plt.plot(thresholds, return_list, label="Return")

    plt.xlabel("Confidence Threshold")
    plt.title("Confidence Sweep")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/plots/confidence_sweep.png")
    plt.close()