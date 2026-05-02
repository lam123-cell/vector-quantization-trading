"""
Generate Freqtrade comparison charts following standard format:
1. Equity Curve
2. Drawdown Curve
3. Trade Distribution
4. Winrate Comparison (Bar Chart)
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


def load_backtest_payload(path: Path) -> dict[str, Any]:
    """Load backtest JSON from ZIP or directly"""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            json_files = [n for n in archive.namelist() 
                         if n.lower().endswith(".json") 
                         and not "config" in n.lower()]
            if not json_files:
                raise ValueError(f"No JSON found in {path}")
            
            main_file = next((f for f in json_files 
                            if "backtest-result" in f.lower()), 
                           json_files[0])
            return json.loads(archive.read(main_file).decode("utf-8"))
    
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class FreqtradeReport:
    """Holds extracted Freqtrade backtest data"""
    strategy_name: str
    trades: pd.DataFrame
    starting_balance: float
    final_balance: float
    total_profit_pct: float
    winrate_pct: float
    num_trades: int
    max_drawdown_pct: float
    profit_factor: float
    trades_df_profit: pd.Series  # profit_abs column for distribution
    
    
def extract_freqtrade_report(payload: dict[str, Any], strategy_name: str = None) -> FreqtradeReport:
    """Extract Freqtrade backtest data"""
    strategies = payload.get("strategy", {})
    
    if strategy_name and strategy_name in strategies:
        strat_data = strategies[strategy_name]
    elif len(strategies) == 1:
        strategy_name = next(iter(strategies.keys()))
        strat_data = strategies[strategy_name]
    else:
        raise ValueError(f"Cannot determine strategy. Available: {list(strategies.keys())}")
    
    trades = strat_data.get("trades", [])
    starting_balance = strat_data.get("starting_balance", 1000)
    final_balance = strat_data.get("final_balance", starting_balance)
    total_profit_pct = strat_data.get("profit_total", 0)
    max_drawdown = strat_data.get("max_drawdown", 0) * 100
    
    num_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get("profit_abs", 0) > 0)
    winrate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    # Profit factor
    gross_profit = sum(max(0, t.get("profit_abs", 0)) for t in trades)
    gross_loss = abs(sum(min(0, t.get("profit_abs", 0)) for t in trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Trades DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return FreqtradeReport(
        strategy_name=strategy_name,
        trades=trades_df,
        starting_balance=starting_balance,
        final_balance=final_balance,
        total_profit_pct=total_profit_pct,
        winrate_pct=winrate,
        num_trades=num_trades,
        max_drawdown_pct=max_drawdown,
        profit_factor=profit_factor,
        trades_df_profit=trades_df["profit_abs"] if "profit_abs" in trades_df.columns else pd.Series([]),
    )


def calculate_equity_curve(trades_df: pd.DataFrame, starting_balance: float) -> tuple[list, list]:
    """
    Calculate equity curve from trades.
    Returns: (timestamps, equity_values)
    """
    if trades_df.empty:
        return [], [starting_balance]
    
    # Sort by close date
    trades_df = trades_df.copy().sort_values("close_date")
    
    equity_values = [starting_balance]
    timestamps = []
    
    current_equity = starting_balance
    
    for _, trade in trades_df.iterrows():
        current_equity += trade.get("profit_abs", 0)
        equity_values.append(current_equity)
        
        # Parse close_date - handle various formats
        close_date = trade.get("close_date", "")
        if close_date:
            timestamps.append(close_date)
    
    return timestamps, equity_values


def calculate_drawdown(equity_values: list) -> list:
    """Calculate running maximum drawdown percentage"""
    equity = pd.Series(equity_values)
    running_max = equity.expanding().max()
    drawdown = ((equity - running_max) / running_max) * 100
    return drawdown.tolist()


# ============ CHART 1: EQUITY CURVE ============
def plot_equity_curve(report: FreqtradeReport, ax=None):
    """Plot equity curve over time"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    timestamps, equity = calculate_equity_curve(report.trades, report.starting_balance)
    
    x = range(len(equity))
    ax.plot(x, equity, linewidth=2, color="steelblue", label="Portfolio Balance")
    ax.axhline(y=report.starting_balance, color="gray", linestyle="--", alpha=0.7, label="Starting Balance")
    ax.fill_between(x, report.starting_balance, equity, alpha=0.3, color="steelblue")
    
    ax.set_xlabel("Trade #", fontsize=12)
    ax.set_ylabel("Portfolio Value (USDT)", fontsize=12)
    ax.set_title(f"{report.strategy_name} - Equity Curve\n"
                f"Total Profit: {report.total_profit_pct:.2f}% | Final: {report.final_balance:.2f} USDT",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


# ============ CHART 2: DRAWDOWN CURVE ============
def plot_drawdown_curve(report: FreqtradeReport, ax=None):
    """Plot drawdown percentage over time"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    timestamps, equity = calculate_equity_curve(report.trades, report.starting_balance)
    drawdown = calculate_drawdown(equity)
    
    x = range(len(drawdown))
    ax.fill_between(x, 0, drawdown, color="red", alpha=0.5, label="Drawdown")
    ax.plot(x, drawdown, color="darkred", linewidth=2)
    ax.axhline(y=report.max_drawdown_pct, color="orange", linestyle="--", linewidth=2, 
              label=f"Max Drawdown: {report.max_drawdown_pct:.2f}%")
    
    ax.set_xlabel("Trade #", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title(f"{report.strategy_name} - Drawdown Curve", fontsize=14, fontweight="bold")
    ax.set_ylim(top=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


# ============ CHART 3: TRADE DISTRIBUTION ============
def plot_trade_distribution(report: FreqtradeReport, ax=None):
    """Plot histogram of trade profits"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if report.trades_df_profit.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        return ax
    
    # Separate wins and losses
    wins = report.trades_df_profit[report.trades_df_profit > 0]
    losses = report.trades_df_profit[report.trades_df_profit < 0]
    
    ax.hist(wins, bins=20, color="green", alpha=0.7, label=f"Wins ({len(wins)})")
    ax.hist(losses, bins=20, color="red", alpha=0.7, label=f"Losses ({len(losses)})")
    
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Profit per Trade (USDT)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{report.strategy_name} - Trade Distribution\n"
                f"Total: {report.num_trades} trades | Winrate: {report.winrate_pct:.1f}%",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    return ax


# ============ CHART 4: COMPARISON METRICS (BAR CHART) ============
def plot_comparison_bars(reports: dict[str, FreqtradeReport], metric_name: str = "Winrate", ax=None):
    """
    Plot bar chart comparing a metric across strategies.
    
    Args:
        reports: dict of {strategy_name: FreqtradeReport}
        metric_name: "Winrate", "Profit Factor", "Profit %", or "Max Drawdown"
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    names = []
    values = []
    
    for name, report in reports.items():
        names.append(name)
        if metric_name == "Winrate":
            values.append(report.winrate_pct)
        elif metric_name == "Profit Factor":
            values.append(report.profit_factor)
        elif metric_name == "Profit %":
            values.append(report.total_profit_pct)
        elif metric_name == "Max Drawdown":
            values.append(abs(report.max_drawdown_pct))
    
    colors = ["green" if metric_name != "Max Drawdown" and v > 0 or metric_name == "Max Drawdown" and v < 0
              else "red" for v in values]
    
    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f"{val:.2f}",
               ha="center", va="bottom" if height > 0 else "top", fontsize=11, fontweight="bold")
    
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Freqtrade Strategy Comparison - {metric_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    return ax


def main():
    parser = argparse.ArgumentParser(description="Generate Freqtrade standard comparison charts")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline backtest result ZIP or JSON")
    parser.add_argument("--turbo", type=Path, required=True, help="Turbo backtest result ZIP or JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for charts")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path("docs/figures") / f"freqtrade_comparison_{date.today().isoformat()}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reports
    print("Loading backtest results...")
    baseline_payload = load_backtest_payload(args.baseline)
    turbo_payload = load_backtest_payload(args.turbo)
    
    baseline_report = extract_freqtrade_report(baseline_payload)
    turbo_report = extract_freqtrade_report(turbo_payload)
    
    print(f"✓ Baseline: {baseline_report.strategy_name} ({baseline_report.num_trades} trades)")
    print(f"✓ Turbo: {turbo_report.strategy_name} ({turbo_report.num_trades} trades)")
    
    # Generate individual charts for each strategy
    for name, report in [("Baseline", baseline_report), ("Turbo", turbo_report)]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        plot_equity_curve(report, ax=axes[0, 0])
        plot_drawdown_curve(report, ax=axes[0, 1])
        plot_trade_distribution(report, ax=axes[1, 0])
        
        # Metrics summary table
        ax_table = axes[1, 1]
        ax_table.axis("off")
        
        metrics_data = [
            ["Metric", "Value"],
            ["Total Profit", f"{report.total_profit_pct:.2f}%"],
            ["Winrate", f"{report.winrate_pct:.1f}%"],
            ["Profit Factor", f"{report.profit_factor:.2f}"],
            ["Max Drawdown", f"{report.max_drawdown_pct:.2f}%"],
            ["Num Trades", f"{report.num_trades}"],
            ["Avg Trade Duration", "TBD"],
        ]
        
        table = ax_table.table(cellText=metrics_data, cellLoc="left", loc="center",
                              colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(2):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")
        
        plt.tight_layout()
        output_file = args.output_dir / f"freqtrade_{name.lower()}_charts.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    # Generate comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    reports_dict = {
        baseline_report.strategy_name: baseline_report,
        turbo_report.strategy_name: turbo_report,
    }
    
    plot_comparison_bars(reports_dict, "Winrate", ax=axes[0, 0])
    plot_comparison_bars(reports_dict, "Profit %", ax=axes[0, 1])
    plot_comparison_bars(reports_dict, "Profit Factor", ax=axes[1, 0])
    plot_comparison_bars(reports_dict, "Max Drawdown", ax=axes[1, 1])
    
    plt.tight_layout()
    output_file = args.output_dir / "freqtrade_comparison_metrics.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    print(f"\n All Freqtrade charts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
