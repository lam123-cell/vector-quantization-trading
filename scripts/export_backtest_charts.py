from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "freqtrade_dataset.csv"
DEFAULT_DATASET_FALLBACK = ROOT / "vq-trading-binance" / "freqtrade_setup" / "user_data" / "freqtrade_dataset.csv"
DEFAULT_RESULTS_ROOT = ROOT / "vq-trading-binance" / "freqtrade_setup" / "user_data" / "backtest_results"
DEFAULT_OUTPUT_DIR = ROOT / "vq-trading-binance" / "docs" / "figures" / f"backtest_report_{date.today().isoformat()}"


@dataclass
class StrategyReport:
    name: str
    trades: pd.DataFrame
    total_trades: int
    profit_total_abs: float
    profit_total_pct: float
    winrate: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_abs: float
    avg_duration: str
    starting_balance: float


def resolve_dataset_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET

    if DEFAULT_DATASET_FALLBACK.exists():
        return DEFAULT_DATASET_FALLBACK

    raise FileNotFoundError("Could not locate freqtrade_dataset.csv")


def resolve_result_path(explicit: str | None, results_root: Path, keyword: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Backtest result not found: {path}")
        return path

    candidates = sorted(
        list(results_root.rglob("*.zip")) + list(results_root.rglob("*.json")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    keyword_lower = keyword.lower()
    for candidate in candidates:
        name_lower = candidate.name.lower()
        if keyword_lower in name_lower:
            return candidate

    raise FileNotFoundError(f"Could not find a backtest result matching '{keyword}' under {results_root}")


def load_json_from_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        json_names = [name for name in archive.namelist() if name.lower().endswith(".json") and not name.lower().endswith("_config.json")]
        if not json_names:
            raise ValueError(f"No JSON payload found inside {zip_path}")

        main_name = next((name for name in json_names if re.search(r"backtest-result-.*\.json$", name)), json_names[0])
        return json.loads(archive.read(main_name).decode("utf-8"))


def load_json_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        return load_json_from_zip(path)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_strategy_block(payload: dict[str, Any], expected_name: str | None = None) -> tuple[str, dict[str, Any]]:
    if "strategy" not in payload:
        raise KeyError("Invalid backtest payload: missing 'strategy'")

    strategy_section = payload["strategy"]
    if not isinstance(strategy_section, dict):
        raise TypeError("Invalid backtest payload: 'strategy' must be a dict")

    if expected_name and expected_name in strategy_section:
        return expected_name, strategy_section[expected_name]

    if len(strategy_section) == 1:
        name = next(iter(strategy_section.keys()))
        return name, strategy_section[name]

    raise KeyError(f"Could not resolve strategy block. Available: {list(strategy_section.keys())}")


def parse_trade_time(trade: dict[str, Any]) -> pd.Timestamp:
    candidates = ["close_date", "close_time", "close_timestamp", "date_close", "close_date_utc"]
    for key in candidates:
        if key not in trade:
            continue

        value = trade[key]
        if value in (None, ""):
            continue

        if isinstance(value, (int, float)):
            unit = "ms" if value > 1e12 else "s"
            return pd.to_datetime(value, unit=unit, utc=True)

        return pd.to_datetime(value, errors="coerce", utc=True)

    raise KeyError(f"No close time field found in trade record: {trade.keys()}")


def parse_trade_profit(trade: dict[str, Any]) -> float:
    for key in ["profit_abs", "profit_total_abs", "profit"]:
        value = trade.get(key)
        if value is not None:
            return float(value)
    return 0.0


def build_trade_frame(strategy_name: str, strategy_block: dict[str, Any]) -> pd.DataFrame:
    trades = strategy_block.get("trades", []) or []
    records = []
    for trade in trades:
        try:
            close_time = parse_trade_time(trade)
        except Exception:
            continue

        records.append(
            {
                "time": close_time,
                "profit_abs": parse_trade_profit(trade),
                "profit_pct": float(trade.get("profit_ratio", trade.get("profit_pct", 0.0))) * 100.0,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    frame = frame.sort_values("time").reset_index(drop=True)
    frame["strategy"] = strategy_name
    return frame


def summarize_strategy(name: str, strategy_block: dict[str, Any]) -> StrategyReport:
    trades_df = build_trade_frame(name, strategy_block)
    starting_balance = float(strategy_block.get("starting_balance", strategy_block.get("dry_run_wallet", 1000.0)))
    total_trades = int(strategy_block.get("total_trades", len(trades_df)))
    profit_total_abs = float(strategy_block.get("profit_total_abs", trades_df["profit_abs"].sum() if not trades_df.empty else 0.0))
    profit_total_pct = float(strategy_block.get("profit_total_pct", 0.0))
    winrate = float(strategy_block.get("winrate", 0.0))

    if not trades_df.empty:
        wins = trades_df[trades_df["profit_abs"] > 0]["profit_abs"].sum()
        losses = trades_df[trades_df["profit_abs"] < 0]["profit_abs"].sum()
        profit_factor = float(wins / abs(losses)) if losses < 0 else math.inf
    else:
        profit_factor = float(strategy_block.get("profit_factor", 0.0))

    if not trades_df.empty:
        equity = starting_balance + trades_df["profit_abs"].cumsum()
        running_max = equity.cummax()
        drawdown_pct = (equity / running_max - 1.0) * 100.0
        drawdown_abs = equity - running_max
        max_drawdown_pct = float(drawdown_pct.min())
        max_drawdown_abs = float(drawdown_abs.min())
    else:
        max_drawdown_pct = float(strategy_block.get("max_relative_drawdown", 0.0))
        max_drawdown_abs = float(strategy_block.get("max_drawdown_abs", 0.0))

    avg_duration = str(strategy_block.get("duration_avg", strategy_block.get("holding_avg", "0:00")))

    return StrategyReport(
        name=name,
        trades=trades_df,
        total_trades=total_trades,
        profit_total_abs=profit_total_abs,
        profit_total_pct=profit_total_pct,
        winrate=winrate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_abs=max_drawdown_abs,
        avg_duration=avg_duration,
        starting_balance=starting_balance,
    )


def daily_equity_series(report: StrategyReport) -> pd.Series:
    if report.trades.empty:
        return pd.Series(dtype=float)

    frame = report.trades.copy()
    frame = frame.set_index("time")
    equity = report.starting_balance + frame["profit_abs"].cumsum()
    daily = equity.resample("1D").last().ffill()
    if daily.empty:
        return equity
    return daily


def daily_drawdown_series(report: StrategyReport) -> pd.Series:
    equity = daily_equity_series(report)
    if equity.empty:
        return equity
    running_max = equity.cummax()
    return (equity / running_max - 1.0) * 100.0


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_path = output_dir / name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_equity_curve(baseline: StrategyReport, turbo: StrategyReport, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))

    for report, color in [(baseline, "#9b2c2c"), (turbo, "#0f766e")]:
        series = daily_equity_series(report)
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=report.name, color=color, linewidth=2.2)

    ax.set_title("Equity Curve Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Balance (USDT)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    return save_figure(fig, output_dir, "01_equity_curve.png")


def plot_drawdown_curve(baseline: StrategyReport, turbo: StrategyReport, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))

    for report, color in [(baseline, "#9b2c2c"), (turbo, "#0f766e")]:
        series = daily_drawdown_series(report)
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=report.name, color=color, linewidth=2.2)

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_title("Drawdown Curve Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    return save_figure(fig, output_dir, "02_drawdown_curve.png")


def plot_distortion_curve(dataset: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))

    data = dataset[["time", "tq_error"]].dropna().copy()
    daily_mean = data.set_index("time")["tq_error"].resample("1D").mean()
    rolling = daily_mean.rolling(7, min_periods=1).mean()

    ax.plot(daily_mean.index, daily_mean.values, color="#6b7280", alpha=0.35, linewidth=1.0, label="Daily mean")
    ax.plot(rolling.index, rolling.values, color="#7c3aed", linewidth=2.2, label="7D rolling mean")
    ax.set_title("Distortion / tq_error Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("tq_error")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    return save_figure(fig, output_dir, "03_tq_error_over_time.png")


def plot_trade_count(baseline: StrategyReport, turbo: StrategyReport, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [baseline.name, turbo.name]
    values = [baseline.total_trades, turbo.total_trades]
    colors = ["#9b2c2c", "#0f766e"]

    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title("Number of Trades Comparison")
    ax.set_ylabel("Trades")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.annotate(f"{value}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=11)

    return save_figure(fig, output_dir, "04_trade_count_comparison.png")


def plot_winrate_profit_factor(baseline: StrategyReport, turbo: StrategyReport, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    labels = [baseline.name, turbo.name]
    colors = ["#9b2c2c", "#0f766e"]

    win_values = [baseline.winrate, turbo.winrate]
    pf_values = [baseline.profit_factor, turbo.profit_factor]

    axes[0].bar(labels, win_values, color=colors, width=0.55)
    axes[0].set_title("Winrate")
    axes[0].set_ylabel("Winrate %")
    axes[0].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(win_values):
        axes[0].annotate(f"{value:.1f}%", (idx, value), ha="center", va="bottom", fontsize=10)

    pf_plot_values = [value if np.isfinite(value) else 0.0 for value in pf_values]
    axes[1].bar(labels, pf_plot_values, color=colors, width=0.55)
    axes[1].set_title("Profit Factor")
    axes[1].set_ylabel("PF")
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(pf_plot_values):
        text = "inf" if not np.isfinite(pf_values[idx]) else f"{value:.2f}"
        axes[1].annotate(text, (idx, value), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Winrate + Profit Factor Comparison", y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir, "05_winrate_profit_factor_comparison.png")


def write_summary(output_dir: Path, baseline: StrategyReport, turbo: StrategyReport, chart_paths: list[Path]) -> None:
    rows = []
    for report in [baseline, turbo]:
        rows.append(
            {
                "strategy": report.name,
                "trades": report.total_trades,
                "profit_total_pct": report.profit_total_pct,
                "profit_total_abs": report.profit_total_abs,
                "winrate": report.winrate,
                "profit_factor": report.profit_factor,
                "max_drawdown_pct": report.max_drawdown_pct,
                "max_drawdown_abs": report.max_drawdown_abs,
                "avg_duration": report.avg_duration,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            if np.isfinite(value):
                return f"{value:.4f}"
            return "inf"
        return str(value)

    headers = list(summary_df.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in summary_df.iterrows():
        table_lines.append("| " + " | ".join(_format_value(row[h]) for h in headers) + " |")

    lines = [
        "# Backtest Chart Export Summary",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        *table_lines,
        "",
        "## Charts",
    ]
    for chart in chart_paths:
        lines.append(f"- {chart.name}")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export five report charts from Freqtrade backtest results.")
    parser.add_argument("--dataset", help="Path to freqtrade_dataset.csv")
    parser.add_argument("--results-root", help="Directory containing Freqtrade backtest result zips", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--baseline-result", help="Baseline backtest result zip/json path")
    parser.add_argument("--turbo-result", help="Turbo backtest result zip/json path")
    parser.add_argument("--output-dir", help="Directory for output images", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--baseline-keyword", default="baseline", help="Keyword used to auto-detect baseline result")
    parser.add_argument("--turbo-keyword", default="turbo", help="Keyword used to auto-detect turbo result")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")

    dataset_path = resolve_dataset_path(args.dataset)
    results_root = Path(args.results_root)
    output_dir = ensure_output_dir(Path(args.output_dir))

    baseline_path = resolve_result_path(args.baseline_result, results_root, args.baseline_keyword)
    turbo_path = resolve_result_path(args.turbo_result, results_root, args.turbo_keyword)

    baseline_payload = load_json_payload(baseline_path)
    turbo_payload = load_json_payload(turbo_path)

    baseline_name, baseline_block = extract_strategy_block(baseline_payload, "FQ_BaselineOHLCVStrategy")
    turbo_name, turbo_block = extract_strategy_block(turbo_payload, "FQ_TurboQuantStrategy")

    baseline = summarize_strategy(baseline_name, baseline_block)
    turbo = summarize_strategy(turbo_name, turbo_block)
    dataset = load_dataset(dataset_path)

    chart_paths = [
        plot_equity_curve(baseline, turbo, output_dir),
        plot_drawdown_curve(baseline, turbo, output_dir),
        plot_distortion_curve(dataset, output_dir),
        plot_trade_count(baseline, turbo, output_dir),
        plot_winrate_profit_factor(baseline, turbo, output_dir),
    ]

    write_summary(output_dir, baseline, turbo, chart_paths)

    print(f"Dataset: {dataset_path}")
    print(f"Baseline result: {baseline_path}")
    print(f"Turbo result: {turbo_path}")
    print(f"Output dir: {output_dir}")
    print("Generated files:")
    for chart in chart_paths:
        print(f"- {chart}")
    print(f"- {output_dir / 'summary_metrics.csv'}")
    print(f"- {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()