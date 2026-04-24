"""
STEP 8 - EVALUATE DRL AGENT

Evaluate a trained PPO model on test data and export key metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.drl_env import TradingEnv


FeatureSet = Literal["baseline", "turbo"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Evaluate PPO model on DRL trading environment"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained PPO model (.zip)",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["baseline", "turbo"],
        required=True,
        help="Feature set to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to test dataset (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiments" / "runs",
        help="Directory to store evaluation outputs",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output filenames",
    )
    return parser.parse_args()


def resolve_dataset_path(feature_set: FeatureSet, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Dataset not found: {explicit}")
        return explicit

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / f"drl_dataset_{feature_set}_test.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run: python scripts/build_dataset.py --mode both"
        )

    return dataset_path


def create_environment(dataset_path: Path, feature_set: FeatureSet) -> TradingEnv:
    if feature_set == "baseline":
        return TradingEnv.from_baseline_csv(dataset_path)
    return TradingEnv.from_turbo_csv(dataset_path)


def compute_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    return float(max_dd)


def compute_sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    if returns.size == 0:
        return 0.0

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    if std_ret < eps:
        return 0.0

    # Per-step Sharpe without annualization, suitable for relative comparison.
    return mean_ret / std_ret


def evaluate_model(model: PPO, env: TradingEnv) -> dict[str, float]:
    obs, _ = env.reset(seed=42)

    done = False
    rewards: list[float] = []
    equity_curve: list[float] = [env.initial_balance]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))

        rewards.append(float(reward))
        equity_curve.append(float(env.get_total_value()))

        done = bool(terminated or truncated)

    env_metrics = env.get_metrics()
    equity = np.asarray(equity_curve, dtype=np.float64)
    step_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)

    total_reward = float(np.sum(rewards))
    final_value = float(env.get_total_value())
    profit = final_value - float(env.initial_balance)
    total_return_pct = (profit / float(env.initial_balance)) * 100.0

    return {
        "total_reward": total_reward,
        "final_portfolio_value": final_value,
        "profit": float(profit),
        "total_return_pct": float(total_return_pct),
        "sharpe_ratio": float(compute_sharpe(step_returns)),
        "max_drawdown": float(compute_max_drawdown(equity_curve)),
        "win_rate": float(env_metrics.get("win_rate", 0.0)),
        "num_trades": float(env_metrics.get("num_trades", 0.0)),
        "win_trades": float(env_metrics.get("win_trades", 0.0)),
        "loss_trades": float(env_metrics.get("loss_trades", 0.0)),
    }


def save_outputs(
    metrics: dict[str, float],
    output_dir: Path,
    feature_set: FeatureSet,
    model_path: Path,
    dataset_path: Path,
    tag: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    stem = f"eval_{feature_set}{suffix}"

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    payload = {
        "feature_set": feature_set,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "metrics": metrics,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    row = {
        "feature_set": feature_set,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        **metrics,
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    return json_path, csv_path


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("STEP 8 - EVALUATE DRL AGENT")
    print("=" * 80)

    try:
        model_path = args.model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        dataset_path = resolve_dataset_path(args.feature_set, args.dataset)

        print(f"[*] Model: {model_path}")
        print(f"[*] Feature set: {args.feature_set}")
        print(f"[*] Dataset: {dataset_path}")

        env = create_environment(dataset_path, args.feature_set)
        model = PPO.load(str(model_path), env=env)

        metrics = evaluate_model(model, env)

        json_path, csv_path = save_outputs(
            metrics=metrics,
            output_dir=args.output_dir,
            feature_set=args.feature_set,
            model_path=model_path,
            dataset_path=dataset_path,
            tag=args.tag,
        )

        print("\n[Evaluation metrics]")
        for key, value in metrics.items():
            if isinstance(value, float) and (not math.isfinite(value)):
                print(f"  - {key}: nan")
            else:
                print(f"  - {key}: {value}")

        print("\n[✓] Evaluation complete!")
        print(f"    JSON: {json_path}")
        print(f"    CSV:  {csv_path}")

        return 0

    except Exception as e:
        print(f"\n[✗] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
