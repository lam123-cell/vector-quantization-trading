#!/usr/bin/env python3
"""
Generate Freqtrade thesis assets:
- 1 comparison table image
- 4 comparison charts

The script is resilient to missing backtest outputs:
- If baseline/turbo result files are available, it renders all charts.
- If not, it still renders the metric table using any available CSV metrics.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.family"] = "DejaVu Sans"


@dataclass
class FreqtradeReport:
    strategy_name: str
    trades: pd.DataFrame
    starting_balance: float
    final_balance: float
    total_profit_pct: float
    winrate_pct: float
    num_trades: int
    max_drawdown_pct: float
    profit_factor: float
    avg_trade_duration_min: float | None


def load_backtest_payload(path: Path) -> dict[str, Any]:
    """Load backtest JSON from ZIP or directly from JSON."""
    if path is None or not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            json_files = [name for name in archive.namelist() if name.lower().endswith(".json") and "config" not in name.lower()]
            if not json_files:
                raise ValueError(f"No JSON found in {path}")

            main_file = next((name for name in json_files if "backtest-result" in name.lower()), json_files[0])
            return json.loads(archive.read(main_file).decode("utf-8"))

    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def auto_find_backtest_file(root: Path, preferred_keyword: str | None = None) -> Path | None:
    """Find a backtest ZIP/JSON if the caller did not pass one explicitly."""
    candidates = []
    for pattern in ["backtest-result-*.zip", "backtest-result-*.json", "*.zip", "*.json"]:
        candidates.extend(root.glob(pattern))

    candidates = [path for path in candidates if path.is_file()]
    if preferred_keyword:
        keyword_matches = [path for path in candidates if preferred_keyword.lower() in path.name.lower()]
        if keyword_matches:
            candidates = keyword_matches

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def auto_find_backtest_file_for_strategy(root: Path, strategy_name: str) -> Path | None:
    """Find the newest backtest file that contains the requested strategy."""
    candidates = []
    for pattern in ["backtest-result-*.zip", "backtest-result-*.json", "*.zip", "*.json"]:
        candidates.extend(root.glob(pattern))

    candidates = [path for path in candidates if path.is_file()]
    candidates = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)

    for path in candidates:
        try:
            payload = load_backtest_payload(path)
        except Exception:
            continue

        strategies = payload.get("strategy", {}) or payload.get("strategy_comparison", {})
        if isinstance(strategies, dict) and strategy_name in strategies:
            return path

        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if isinstance(metadata, dict) and metadata.get("strategy") == strategy_name:
            return path

    return None


def parse_datetime(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None)


def estimate_trade_duration_minutes(trade: dict[str, Any]) -> float | None:
    """Estimate trade duration in minutes from available trade fields."""
    for key in [
        "trade_duration",
        "trade_duration_s",
        "trade_duration_secs",
        "trade_duration_seconds",
        "trade_duration_min",
        "trade_duration_minutes",
    ]:
        if key in trade and trade[key] not in (None, ""):
            try:
                value = float(trade[key])
                if "_s" in key or "sec" in key:
                    return value / 60.0
                return value
            except (TypeError, ValueError):
                pass

    open_date = parse_datetime(trade.get("open_date") or trade.get("open_timestamp") or trade.get("open_time"))
    close_date = parse_datetime(trade.get("close_date") or trade.get("close_timestamp") or trade.get("close_time"))
    if open_date is not None and close_date is not None and close_date >= open_date:
        return (close_date - open_date).total_seconds() / 60.0
    return None


def safe_mean(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def extract_drawdown_pct(strategy_data: dict[str, Any], trades_df: pd.DataFrame | None = None) -> float:
    """Extract max drawdown as a percentage from the best available source."""
    for key in ("max_relative_drawdown", "max_drawdown_account", "max_drawdown"):
        value = strategy_data.get(key)
        if value is None or value == "":
            continue
        try:
            drawdown = float(value)
        except (TypeError, ValueError):
            continue
        if key == "max_drawdown":
            return drawdown * 100.0 if abs(drawdown) <= 1.0 else drawdown
        return drawdown * 100.0 if abs(drawdown) <= 1.0 else drawdown

    if trades_df is not None and not trades_df.empty and "profit_abs" in trades_df.columns:
        equity_series = 1000.0 + trades_df["profit_abs"].cumsum().astype(float)
        peak = equity_series.cummax()
        drawdowns = ((equity_series - peak) / peak.replace(0, np.nan)) * 100.0
        if not drawdowns.empty and not pd.isna(drawdowns.min()):
            return abs(float(drawdowns.min()))

    return 0.0


def extract_freqtrade_report(payload: dict[str, Any], strategy_name: str | None = None) -> FreqtradeReport:
    """Extract a single Freqtrade strategy report from backtest payload.
    
    Supports two formats:
    1. Backtest ZIP format: {"strategy": {strategy_name: {...}}}
    2. Trades JSON format: {"trades": [...], "metadata": {...}}
    """
    # Handle trades.json format (only has "trades" key)
    if "trades" in payload and "strategy" not in payload and "strategy_comparison" not in payload:
        trades = payload.get("trades", []) or []
        metadata = payload.get("metadata", {})
        strat_name = metadata.get("strategy", "Unknown Strategy")
        
        # Calculate metrics directly from trades
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            if "close_date" in trades_df.columns:
                trades_df["_close_dt"] = pd.to_datetime(trades_df["close_date"], errors="coerce", utc=True).dt.tz_convert(None)
                trades_df = trades_df.sort_values("_close_dt", na_position="last")
            elif "close_time" in trades_df.columns:
                trades_df["_close_dt"] = pd.to_datetime(trades_df["close_time"], errors="coerce", utc=True).dt.tz_convert(None)
                trades_df = trades_df.sort_values("_close_dt", na_position="last")
        
        # Ensure profit_abs is numeric
        if not trades_df.empty and "profit_abs" in trades_df.columns:
            trades_df["profit_abs"] = pd.to_numeric(trades_df["profit_abs"], errors="coerce").fillna(0.0)
        
        starting_balance = 1000.0  # Default if not in trades.json
        total_profit_usdt = sum(float(trade.get("profit_abs", 0.0)) for trade in trades)
        final_balance = starting_balance + total_profit_usdt
        total_profit_pct = (total_profit_usdt / starting_balance * 100.0) if starting_balance > 0 else 0.0
        
        num_trades = len(trades)
        winning_trades = sum(1 for trade in trades if float(trade.get("profit_abs", 0.0)) > 0)
        winrate_pct = (winning_trades / num_trades * 100.0) if num_trades > 0 else 0.0
        
        gross_profit = sum(max(0.0, float(trade.get("profit_abs", 0.0))) for trade in trades)
        gross_loss = abs(sum(min(0.0, float(trade.get("profit_abs", 0.0))) for trade in trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        avg_trade_duration_min = safe_mean([estimate_trade_duration_minutes(trade) for trade in trades])
        
        # Calculate max drawdown from cumulative profit
        equity_series = starting_balance + trades_df["profit_abs"].cumsum().astype(float)
        peak = equity_series.cummax()
        drawdowns = ((equity_series - peak) / peak.replace(0, np.nan)) * 100.0
        max_drawdown_pct = abs(drawdowns.min()) if not drawdowns.empty else 0.0
        
        return FreqtradeReport(
            strategy_name=strat_name,
            trades=trades_df,
            starting_balance=starting_balance,
            final_balance=final_balance,
            total_profit_pct=total_profit_pct,
            winrate_pct=winrate_pct,
            num_trades=num_trades,
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            avg_trade_duration_min=avg_trade_duration_min,
        )
    
    # Handle backtest ZIP format
    strategies = payload.get("strategy", {}) or payload.get("strategy_comparison", {})
    if not strategies:
        raise ValueError("No strategy data found in backtest payload")

    if strategy_name and strategy_name in strategies:
        strat_name = strategy_name
        strat_data = strategies[strategy_name]
    elif len(strategies) == 1:
        strat_name = next(iter(strategies.keys()))
        strat_data = strategies[strat_name]
    else:
        strat_name = next(iter(strategies.keys()))
        strat_data = strategies[strat_name]

    trades = strat_data.get("trades", []) or []
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        if "close_date" in trades_df.columns:
            trades_df["_close_dt"] = pd.to_datetime(trades_df["close_date"], errors="coerce", utc=True).dt.tz_convert(None)
            trades_df = trades_df.sort_values("_close_dt", na_position="last")
        elif "close_time" in trades_df.columns:
            trades_df["_close_dt"] = pd.to_datetime(trades_df["close_time"], errors="coerce", utc=True).dt.tz_convert(None)
            trades_df = trades_df.sort_values("_close_dt", na_position="last")

    starting_balance = float(strat_data.get("starting_balance", 1000))
    final_balance = float(strat_data.get("final_balance", starting_balance))
    total_profit_pct = float(strat_data.get("profit_total", strat_data.get("profit_total_pct", 0.0)))
    max_drawdown_pct = extract_drawdown_pct(strat_data, trades_df)

    num_trades = len(trades)
    winning_trades = sum(1 for trade in trades if float(trade.get("profit_abs", 0.0)) > 0)
    winrate_pct = (winning_trades / num_trades * 100.0) if num_trades > 0 else 0.0

    gross_profit = sum(max(0.0, float(trade.get("profit_abs", 0.0))) for trade in trades)
    gross_loss = abs(sum(min(0.0, float(trade.get("profit_abs", 0.0))) for trade in trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_trade_duration_min = safe_mean([estimate_trade_duration_minutes(trade) for trade in trades])

    if not trades_df.empty and "profit_abs" in trades_df.columns:
        trades_df["profit_abs"] = pd.to_numeric(trades_df["profit_abs"], errors="coerce").fillna(0.0)

    return FreqtradeReport(
        strategy_name=strat_name,
        trades=trades_df,
        starting_balance=starting_balance,
        final_balance=final_balance,
        total_profit_pct=total_profit_pct,
        winrate_pct=winrate_pct,
        num_trades=num_trades,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        avg_trade_duration_min=avg_trade_duration_min,
    )


def build_equity_frame(report: FreqtradeReport) -> pd.DataFrame:
    if report.trades.empty or "profit_abs" not in report.trades.columns:
        return pd.DataFrame(columns=["time", "equity", "drawdown"])

    trades_df = report.trades.copy()
    if "_close_dt" in trades_df.columns:
        time_series = pd.to_datetime(trades_df["_close_dt"], errors="coerce")
    elif "close_date" in trades_df.columns:
        time_series = pd.to_datetime(trades_df["close_date"], errors="coerce", utc=True).dt.tz_convert(None)
    elif "close_time" in trades_df.columns:
        time_series = pd.to_datetime(trades_df["close_time"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        time_series = pd.Series(pd.RangeIndex(len(trades_df)))

    equity = report.starting_balance + trades_df["profit_abs"].cumsum().astype(float)
    peak = equity.cummax()
    drawdown = ((equity - peak) / peak.replace(0, np.nan)) * 100.0

    return pd.DataFrame(
        {
            "time": time_series,
            "equity": equity,
            "drawdown": drawdown.fillna(0.0),
        }
    ).dropna(subset=["time"])





def fmt_float(value: float | None, suffix: str = "", digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def fmt_money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:.2f}"


def _improvement_scalar(base: float | None, turbo: float | None, lower_is_better: bool = False, suffix: str = "") -> str:
    if base is None or turbo is None or pd.isna(base) or pd.isna(turbo):
        return "-"
    delta = turbo - base
    if lower_is_better:
        delta = base - turbo
    return f"{delta:+.2f}{suffix}"


def _improvement_pct(base: float | None, turbo: float | None, lower_is_better: bool = False) -> str:
    if base is None or turbo is None or pd.isna(base) or pd.isna(turbo):
        return "-"
    delta = turbo - base
    if lower_is_better:
        delta = base - turbo
    return f"{delta:+.2f}pp"


def _improvement_money(base: float | None, turbo: float | None) -> str:
    if base is None or turbo is None or pd.isna(base) or pd.isna(turbo):
        return "-"
    return f"${turbo - base:+.2f}"


def create_metrics_table_image(baseline_report: FreqtradeReport | None, turbo_report: FreqtradeReport | None, output_image: Path) -> None:
    """Create a clean comparison table for thesis use."""
    rows = [
        ["Metric", "Baseline", "Turbo", "Improvement"],
        ["Total Profit (%)",
         fmt_float(baseline_report.total_profit_pct if baseline_report else None, "%", 2),
         fmt_float(turbo_report.total_profit_pct if turbo_report else None, "%", 2),
         _improvement_pct(baseline_report.total_profit_pct if baseline_report else None,
                          turbo_report.total_profit_pct if turbo_report else None)],
        ["Max Drawdown (%)",
         fmt_float(baseline_report.max_drawdown_pct if baseline_report else None, "%", 2),
         fmt_float(turbo_report.max_drawdown_pct if turbo_report else None, "%", 2),
         _improvement_pct(baseline_report.max_drawdown_pct if baseline_report else None,
                          turbo_report.max_drawdown_pct if turbo_report else None,
                          lower_is_better=True)],
        ["Final Portfolio Value",
         fmt_money(baseline_report.final_balance if baseline_report else None),
         fmt_money(turbo_report.final_balance if turbo_report else None),
         _improvement_money(baseline_report.final_balance if baseline_report else None,
                            turbo_report.final_balance if turbo_report else None)],
        ["Trades",
         fmt_float(baseline_report.num_trades if baseline_report else None, digits=0),
         fmt_float(turbo_report.num_trades if turbo_report else None, digits=0),
         _improvement_scalar(baseline_report.num_trades if baseline_report else None,
                             turbo_report.num_trades if turbo_report else None)],
        ["Winrate",
         fmt_float(baseline_report.winrate_pct if baseline_report else None, "%", 2),
         fmt_float(turbo_report.winrate_pct if turbo_report else None, "%", 2),
         _improvement_pct(baseline_report.winrate_pct if baseline_report else None,
                          turbo_report.winrate_pct if turbo_report else None)],
        ["Profit Factor",
         fmt_float(baseline_report.profit_factor if baseline_report else None, digits=2),
         fmt_float(turbo_report.profit_factor if turbo_report else None, digits=2),
         _improvement_scalar(baseline_report.profit_factor if baseline_report else None,
                             turbo_report.profit_factor if turbo_report else None)],
        ["Avg Trade Duration",
         fmt_float(baseline_report.avg_trade_duration_min if baseline_report else None, " min", 2),
         fmt_float(turbo_report.avg_trade_duration_min if turbo_report else None, " min", 2),
         _improvement_scalar(baseline_report.avg_trade_duration_min if baseline_report else None,
                             turbo_report.avg_trade_duration_min if turbo_report else None,
                             lower_is_better=True,
                             suffix=" min")],
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    table = ax.table(cellText=rows, cellLoc="center", loc="center", colWidths=[0.28, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for i in range(4):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(weight="bold", color="white", fontsize=11)
        table[(0, i)].set_edgecolor("white")
        table[(0, i)].set_linewidth(1.5)

    for row_idx in range(1, len(rows)):
        bg = "#f7f7f7" if row_idx % 2 == 0 else "white"
        for col_idx in range(4):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#d5d8dc")
            cell.set_linewidth(0.6)
            cell.set_text_props(fontsize=10, color="black")

        for col_idx in (1, 2):
            text = str(rows[row_idx][col_idx])
            if text.startswith("-"):
                table[(row_idx, col_idx)].set_text_props(color="#e74c3c", weight="bold")
            elif text.startswith("+"):
                table[(row_idx, col_idx)].set_text_props(color="#27ae60", weight="bold")

        improvement_text = str(rows[row_idx][3])
        if improvement_text != "-":
            table[(row_idx, 3)].set_facecolor("#d5f5e3")
            table[(row_idx, 3)].set_text_props(color="#1e8449", weight="bold")
            table[(row_idx, 3)].get_text().set_text(f"✓ {improvement_text}")

    fig.text(0.5, 0.97, "Freqtrade - Bảng so sánh Baseline vs Turbo", ha="center", fontsize=18, fontweight="bold", color="#1f2a44")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05)
    fig.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve_comparison(baseline_report: FreqtradeReport, turbo_report: FreqtradeReport, output_image: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = False

    for report, color, label in [
        (baseline_report, "steelblue", "Baseline"),
        (turbo_report, "darkorange", "Turbo"),
    ]:
        frame = build_equity_frame(report)
        if frame.empty:
            continue
        plotted = True
        ax.plot(frame["time"], frame["equity"], linewidth=2.2, label=label, color=color)
        ax.fill_between(frame["time"], report.starting_balance, frame["equity"], alpha=0.14, color=color)

    if not plotted:
        ax.text(0.5, 0.5, "No equity data found", transform=ax.transAxes, ha="center", va="center", fontsize=14)
    else:
        ax.axhline(y=baseline_report.starting_balance, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, label="Starting Balance")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Balance (USDT)", fontsize=12)
    ax.set_title("Freqtrade - Equity Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown_curve_comparison(baseline_report: FreqtradeReport, turbo_report: FreqtradeReport, output_image: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = False

    for report, color, label in [
        (baseline_report, "crimson", "Baseline"),
        (turbo_report, "darkgreen", "Turbo"),
    ]:
        frame = build_equity_frame(report)
        if frame.empty:
            continue
        plotted = True
        ax.fill_between(frame["time"], 0, frame["drawdown"], alpha=0.18, color=color)
        ax.plot(frame["time"], frame["drawdown"], linewidth=2.0, label=label, color=color)

    if not plotted:
        ax.text(0.5, 0.5, "No drawdown data found", transform=ax.transAxes, ha="center", va="center", fontsize=14)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title("Freqtrade - Drawdown Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trade_distribution_comparison(baseline_report: FreqtradeReport, turbo_report: FreqtradeReport, output_image: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    baseline_profit = baseline_report.trades["profit_abs"] if not baseline_report.trades.empty and "profit_abs" in baseline_report.trades.columns else pd.Series(dtype=float)
    turbo_profit = turbo_report.trades["profit_abs"] if not turbo_report.trades.empty and "profit_abs" in turbo_report.trades.columns else pd.Series(dtype=float)

    if baseline_profit.empty and turbo_profit.empty:
        ax.text(0.5, 0.5, "No trade distribution data found", transform=ax.transAxes, ha="center", va="center", fontsize=14)
    else:
        combined = pd.concat([baseline_profit, turbo_profit], ignore_index=True)
        bins = np.histogram_bin_edges(combined.dropna(), bins=20) if not combined.dropna().empty else 20

        if not baseline_profit.empty:
            ax.hist(baseline_profit.dropna(), bins=bins, alpha=0.55, color="steelblue", label=f"Baseline ({len(baseline_profit)})")
        if not turbo_profit.empty:
            ax.hist(turbo_profit.dropna(), bins=bins, alpha=0.55, color="darkorange", label=f"Turbo ({len(turbo_profit)})")

        ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--")

    ax.set_xlabel("Profit per Trade (USDT)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Freqtrade - Trade Distribution", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    fig.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_winrate_comparison(baseline_report: FreqtradeReport, turbo_report: FreqtradeReport, output_image: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ["Baseline", "Turbo"]
    values = [baseline_report.winrate_pct, turbo_report.winrate_pct]
    colors = ["steelblue", "darkorange"]
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Winrate (%)", fontsize=12)
    ax.set_title("Freqtrade - Winrate Comparison", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    fig.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(path: Path | None, strategy_name: str | None = None) -> FreqtradeReport | None:
    if path is None or not path.exists():
        return None
    payload = load_backtest_payload(path)
    return extract_freqtrade_report(payload, strategy_name=strategy_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Freqtrade comparison table and charts")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline backtest result ZIP/JSON")
    parser.add_argument("--turbo", type=Path, default=None, help="Turbo backtest result ZIP/JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for the images")
    parser.add_argument("--baseline-strategy", type=str, default=None, help="Optional baseline strategy name")
    parser.add_argument("--turbo-strategy", type=str, default=None, help="Optional turbo strategy name")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path("docs/figures") / f"freqtrade_comparison_{date.today().isoformat()}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    freqtrade_root = Path("freqtrade_setup/user_data/backtest_results")
    baseline_path = args.baseline or auto_find_backtest_file_for_strategy(freqtrade_root, args.baseline_strategy or "FQ_BaselineFairStrategy")
    turbo_path = args.turbo or auto_find_backtest_file_for_strategy(freqtrade_root, args.turbo_strategy or "FQ_TurboCoreFairStrategy")

    if baseline_path is None:
        baseline_path = auto_find_backtest_file(freqtrade_root, "baseline")
    if turbo_path is None:
        turbo_path = auto_find_backtest_file(freqtrade_root, "turbo")

    if baseline_path == turbo_path:
        ordered = sorted(
            [path for path in freqtrade_root.glob("backtest-result-*.zip") if path.is_file()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if len(ordered) >= 2:
            baseline_path = ordered[1]
            turbo_path = ordered[0]

    baseline_report = build_report(baseline_path, args.baseline_strategy) if baseline_path else None
    turbo_report = build_report(turbo_path, args.turbo_strategy) if turbo_path else None

    table_path = args.output_dir / "freqtrade_metrics_table.png"
    create_metrics_table_image(baseline_report, turbo_report, table_path)
    print(f"✓ Metrics table saved: {table_path}")

    if baseline_report is None or turbo_report is None:
        print("[!] No freqtrade backtest result ZIP/JSON found yet, so chart generation was skipped.")
        print("    Put backtest-result-*.zip files in freqtrade_setup/user_data/backtest_results/ or pass --baseline/--turbo.")
        return

    plot_equity_curve_comparison(baseline_report, turbo_report, args.output_dir / "freqtrade_equity_curve.png")
    plot_drawdown_curve_comparison(baseline_report, turbo_report, args.output_dir / "freqtrade_drawdown_curve.png")
    plot_trade_distribution_comparison(baseline_report, turbo_report, args.output_dir / "freqtrade_trade_distribution.png")
    plot_winrate_comparison(baseline_report, turbo_report, args.output_dir / "freqtrade_winrate_comparison.png")

    print(f"✓ Charts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()