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


def evaluate_model(model: PPO, env: TradingEnv) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    obs, _ = env.reset(seed=42)

    done = False
    rewards: list[float] = []
    equity_curve: list[float] = [env.initial_balance]
    action_counts = {0: 0, 1: 0, 2: 0}
    peak_equity = float(env.initial_balance)
    action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}
    rollout_rows: list[dict[str, float | int | str | bool]] = []

    while not done:
        step_before = int(env.current_step)
        row_before = env.data.iloc[step_before]
        time_before = str(row_before.get("time", ""))
        close_before = float(row_before["close"])
        balance_before = float(env.balance)
        position_before = float(env.position)

        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        action_counts[action_int] = action_counts.get(action_int, 0) + 1
        obs, reward, terminated, truncated, _ = env.step(action_int)

        rewards.append(float(reward))
        equity_after = float(env.get_total_value())
        equity_curve.append(equity_after)
        peak_equity = max(peak_equity, equity_after)
        drawdown = (peak_equity - equity_after) / peak_equity if peak_equity > 0 else 0.0

        rollout_rows.append(
            {
                "step": step_before,
                "time": time_before,
                "close": close_before,
                "action": action_int,
                "action_name": action_name.get(action_int, "UNKNOWN"),
                "reward": float(reward),
                "balance_before": balance_before,
                "position_before": position_before,
                "balance_after": float(env.balance),
                "position_after": float(env.position),
                "equity": equity_after,
                "drawdown": float(drawdown),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        done = bool(terminated or truncated)

    env_metrics = env.get_metrics()
    equity = np.asarray(equity_curve, dtype=np.float64)
    step_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)

    total_reward = float(np.sum(rewards))
    final_value = float(env.get_total_value())
    profit = final_value - float(env.initial_balance)
    total_return_pct = (profit / float(env.initial_balance)) * 100.0
    total_steps = float(sum(action_counts.values()))
    hold_count = float(action_counts.get(0, 0))
    buy_count = float(action_counts.get(1, 0))
    sell_count = float(action_counts.get(2, 0))

    metrics = {
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
        "eval_total_steps": total_steps,
        "action_hold_count": hold_count,
        "action_buy_count": buy_count,
        "action_sell_count": sell_count,
        "action_hold_ratio": (hold_count / total_steps) if total_steps > 0 else 0.0,
        "action_buy_ratio": (buy_count / total_steps) if total_steps > 0 else 0.0,
        "action_sell_ratio": (sell_count / total_steps) if total_steps > 0 else 0.0,
    }

    rollout_df = pd.DataFrame(rollout_rows)
    trades_df = pd.DataFrame(env.trades)

    return metrics, rollout_df, trades_df


def save_outputs(
    metrics: dict[str, float],
    rollout_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: Path,
    feature_set: FeatureSet,
    model_path: Path,
    dataset_path: Path,
    tag: str,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    stem = f"eval_{feature_set}{suffix}"

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    rollout_path = output_dir / f"{stem}_rollout.csv"
    trades_path = output_dir / f"{stem}_trades.csv"

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
    rollout_df.to_csv(rollout_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    # Also export a simplified eval results CSV (time, balance/equity) for plotting
    try:
        eval_results_path = output_dir / f"{stem}_results.csv"
        if "time" in rollout_df.columns and "equity" in rollout_df.columns:
            rollout_df[["time", "equity"]].to_csv(eval_results_path, index=False)
        elif "time" in rollout_df.columns and "balance_after" in rollout_df.columns:
            rollout_df[["time", "balance_after"]].rename(columns={"balance_after": "equity"}).to_csv(eval_results_path, index=False)
        else:
            # fallback: save step and equity/balance
            if "equity" in rollout_df.columns:
                rollout_df[["step", "equity"]].to_csv(eval_results_path, index=False)
        # attempt copy to Drive (colab)
        try:
            import shutil
            drive_path = Path("/content/drive/MyDrive/thesis")
            drive_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(eval_results_path), str(drive_path / eval_results_path.name))
            print(f"[*] Copied eval results to Drive: {drive_path / eval_results_path.name}")
        except Exception:
            print("[!] Drive copy for eval results skipped or failed")
    except Exception:
        pass

    return json_path, csv_path, rollout_path, trades_path


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

        metrics, rollout_df, trades_df = evaluate_model(model, env)

        json_path, csv_path, rollout_path, trades_path = save_outputs(
            metrics=metrics,
            rollout_df=rollout_df,
            trades_df=trades_df,
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
        print(f"    Rollout CSV: {rollout_path}")
        print(f"    Trades CSV:  {trades_path}")

        if metrics.get("action_buy_count", 0.0) == 0.0 and metrics.get("action_sell_count", 0.0) == 0.0:
            print("\n[!] Policy diagnostic: model predicted HOLD for all evaluation steps.")
            print("    This usually means the model was trained with old environment logic")
            print("    or Colab is still using an outdated copy of src/models/drl_env.py.")

        return 0

    except Exception as e:
        print(f"\n[✗] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
