import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.plot_utils import (
    plot_compare_classification,
    plot_confidence_distribution,
    plot_confusion,
    plot_prediction_distribution,
    save_lstm_metrics_table,
)


RESULT_DIR = "results"


def load_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file: {path}")
    return np.load(path)


def load_optional_latency(path):
    if not os.path.exists(path):
        return 0.0
    return float(np.load(path)[0])


def get_metrics(y_true, y_pred, confs, latency=0.0):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "avg_confidence": float(np.mean(confs)),
        "latency": latency,
    }


def main():
    base_preds = load_array(f"{RESULT_DIR}/baseline_preds.npy")
    base_trues = load_array(f"{RESULT_DIR}/baseline_trues.npy")
    base_confs = load_array(f"{RESULT_DIR}/baseline_confs.npy")

    tq_preds = load_array(f"{RESULT_DIR}/tq_preds.npy")
    tq_trues = load_array(f"{RESULT_DIR}/tq_trues.npy")
    tq_confs = load_array(f"{RESULT_DIR}/tq_confs.npy")
    base_latency = load_optional_latency(f"{RESULT_DIR}/baseline_latency.npy")
    tq_latency = load_optional_latency(f"{RESULT_DIR}/tq_latency.npy")

    base_m = get_metrics(base_trues, base_preds, base_confs, latency=base_latency)
    tq_m = get_metrics(tq_trues, tq_preds, tq_confs, latency=tq_latency)

    plot_compare_classification(base_m, tq_m)
    save_lstm_metrics_table(base_m, tq_m)
    plot_prediction_distribution(base_trues, base_preds, tq_preds)
    plot_confidence_distribution(base_confs, tq_confs)
    plot_confusion(base_trues, base_preds, "Baseline LSTM")
    plot_confusion(tq_trues, tq_preds, "TurboQuant LSTM")

    print("\n===== LSTM COMPARISON =====")
    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "avg_confidence",
        "latency",
    ]:
        print(f"{key:18} | baseline: {base_m[key]:.6f} | turboquant: {tq_m[key]:.6f}")

    print("\nPlots saved to: results/plots")


if __name__ == "__main__":
    main()
