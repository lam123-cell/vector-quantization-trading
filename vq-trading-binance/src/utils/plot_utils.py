import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


CLASS_IDS = [0, 1, 2]
CLASS_LABELS = ["Sell", "Hold", "Buy"]
PLOT_DIR = "results/plots"


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_compare_classification(base_m, tq_m):
    ensure_plot_dir()

    labels = [
        "Accuracy",
        "Macro F1",
        "Macro Precision",
        "Macro Recall",
        "Balanced Acc.",
    ]
    base_vals = [
        base_m["accuracy"],
        base_m["macro_f1"],
        base_m["macro_precision"],
        base_m["macro_recall"],
        base_m["balanced_accuracy"],
    ]
    tq_vals = [
        tq_m["accuracy"],
        tq_m["macro_f1"],
        tq_m["macro_precision"],
        tq_m["macro_recall"],
        tq_m["balanced_accuracy"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, base_vals, width, label="Baseline")
    plt.bar(x + width / 2, tq_vals, width, label="TurboQuant")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("LSTM Classification Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/lstm_classification_metrics.png", dpi=300)
    plt.close()


def save_lstm_metrics_table(base_m, tq_m):
    ensure_plot_dir()

    rows = [
        ("Accuracy", base_m["accuracy"], tq_m["accuracy"]),
        ("Macro F1", base_m["macro_f1"], tq_m["macro_f1"]),
        ("Macro Precision", base_m["macro_precision"], tq_m["macro_precision"]),
        ("Macro Recall", base_m["macro_recall"], tq_m["macro_recall"]),
        ("Balanced Accuracy", base_m["balanced_accuracy"], tq_m["balanced_accuracy"]),
        ("Avg Confidence", base_m["avg_confidence"], tq_m["avg_confidence"]),
        ("Latency (sec/sample)", base_m["latency"], tq_m["latency"]),
    ]

    df = pd.DataFrame(
        {
            "Metric": [row[0] for row in rows],
            "Baseline": [f"{row[1]:.6f}" if "Latency" in row[0] else f"{row[1]:.4f}" for row in rows],
            "TurboQuant": [f"{row[2]:.6f}" if "Latency" in row[0] else f"{row[2]:.4f}" for row in rows],
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.35)
    plt.title("LSTM Baseline vs TurboQuant Summary", pad=18)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/lstm_metrics_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, tag):
    ensure_plot_dir()
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)

    plt.figure(figsize=(5.6, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{tag} Confusion Matrix")
    plt.tight_layout()
    filename = tag.lower().replace(" ", "_")
    plt.savefig(f"{PLOT_DIR}/{filename}_confusion.png", dpi=300)
    plt.close()


def plot_prediction_distribution(y_true, base_preds, tq_preds):
    ensure_plot_dir()

    true_counts = np.bincount(y_true, minlength=3)
    base_counts = np.bincount(base_preds, minlength=3)
    tq_counts = np.bincount(tq_preds, minlength=3)

    x = np.arange(len(CLASS_LABELS))
    width = 0.25

    plt.figure(figsize=(8.4, 4.6))
    plt.bar(x - width, true_counts, width, label="True")
    plt.bar(x, base_counts, width, label="Baseline")
    plt.bar(x + width, tq_counts, width, label="TurboQuant")
    plt.xticks(x, CLASS_LABELS)
    plt.ylabel("Number of samples")
    plt.title("LSTM Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/lstm_prediction_distribution.png", dpi=300)
    plt.close()


def plot_confidence_distribution(base_confs, tq_confs):
    ensure_plot_dir()

    plt.figure(figsize=(7.6, 4.6))
    plt.hist(base_confs, bins=30, alpha=0.65, label="Baseline")
    plt.hist(tq_confs, bins=30, alpha=0.65, label="TurboQuant")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of samples")
    plt.title("LSTM Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/lstm_confidence_distribution.png", dpi=300)
    plt.close()
