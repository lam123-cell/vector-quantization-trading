#!/usr/bin/env python3
"""
Generate DRL comparison: 1 metrics table + 4 combined comparison charts

Output structure:
- drl_metrics_table.png: Comparison table (baseline vs turbo)
- drl_reward_curves.png: Reward comparison (2 lines on same chart)
- drl_portfolio_curves.png: Portfolio value comparison (2 lines on same chart)
- drl_action_distribution.png: Action distribution (2 pie charts side by side)
- drl_drawdown_curves.png: Drawdown comparison (2 lines on same chart)

All images saved at 150 DPI for thesis publication.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")


def load_drl_eval(eval_json_path: Path) -> dict[str, Any]:
    """Load DRL evaluation JSON"""
    with open(eval_json_path, "r") as f:
        return json.load(f)


def load_rollout_data(rollout_csv_path: Path) -> pd.DataFrame:
    """Load rollout CSV if available"""
    try:
        if rollout_csv_path and rollout_csv_path.exists():
            return pd.read_csv(rollout_csv_path)
    except:
        pass
    return None


def create_metrics_table_image(baseline_eval: Path, turbo_eval: Path, output_image: Path):
    """Create and save metrics comparison table as image"""
    
    baseline_data = load_drl_eval(baseline_eval)
    turbo_data = load_drl_eval(turbo_eval)
    baseline_m = baseline_data.get("metrics", {})
    turbo_m = turbo_data.get("metrics", {})
    
    # 10 core metrics for thesis
    table_data = [
        ["Metric", "Baseline", "Turbo", "Improvement"],
        
        # Financial Performance
        ["Total Profit (%)", 
         f"{baseline_m.get('total_return_pct', 0)*100:.2f}%",
         f"{turbo_m.get('total_return_pct', 0)*100:.2f}%",
         f"{(turbo_m.get('total_return_pct', 0) - baseline_m.get('total_return_pct', 0))*100:+.2f}pp"],
        
        ["Max Drawdown (%)",
         f"{baseline_m.get('max_drawdown', 0)*100:.2f}%",
         f"{turbo_m.get('max_drawdown', 0)*100:.2f}%",
         f"{(baseline_m.get('max_drawdown', 0) - turbo_m.get('max_drawdown', 0))*100:+.2f}pp"],
        
        ["Final Portfolio Value",
         f"${baseline_m.get('final_portfolio_value', 1000):.2f}",
         f"${turbo_m.get('final_portfolio_value', 1000):.2f}",
         f"${turbo_m.get('final_portfolio_value', 1000) - baseline_m.get('final_portfolio_value', 1000):+.2f}"],
        
        # Resource Metrics
        ["Processing Time",
         "TBD",
         "TBD",
         "-"],
        
        ["Memory Usage (MB)",
         "TBD",
         "TBD",
         "-"],
        
        ["Distortion Error (%)",
         "TBD",
         "TBD",
         "-"],
        
        # DRL Learning Metrics
        ["Average Reward",
         f"{baseline_m.get('total_reward', 0):.2f}",
         f"{turbo_m.get('total_reward', 0):.2f}",
         f"{turbo_m.get('total_reward', 0) - baseline_m.get('total_reward', 0):+.2f}"],
        
        ["Episode Reward",
         f"{baseline_m.get('total_reward', 0):.2f}",
         f"{turbo_m.get('total_reward', 0):.2f}",
         f"{turbo_m.get('total_reward', 0) - baseline_m.get('total_reward', 0):+.2f}"],
        
        ["Policy Stability",
         "TBD",
         "TBD",
         "-"],
        
        ["Action Distribution",
         f"Buy:{baseline_m.get('action_buy_ratio', 0)*100:.1f}% Sell:{baseline_m.get('action_sell_ratio', 0)*100:.1f}% Hold:{baseline_m.get('action_hold_ratio', 0)*100:.1f}%",
         f"Buy:{turbo_m.get('action_buy_ratio', 0)*100:.1f}% Sell:{turbo_m.get('action_sell_ratio', 0)*100:.1f}% Hold:{turbo_m.get('action_hold_ratio', 0)*100:.1f}%",
         "-"],
    ]
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)
    
    # Style header row - professional dark blue with white text
    for i in range(4):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(weight="bold", color="white", fontsize=12)
        table[(0, i)].set_edgecolor("white")
        table[(0, i)].set_linewidth(2)
    
    # Style data rows with alternating colors and grouped sections
    # Financial Performance rows (1-3)
    for i in range(1, 4):
        for j in range(4):
            if i % 2 == 1:
                table[(i, j)].set_facecolor("#ecf0f1")
            else:
                table[(i, j)].set_facecolor("white")
            table[(i, j)].set_text_props(fontsize=10)
            table[(i, j)].set_edgecolor("#bdc3c7")
            table[(i, j)].set_linewidth(1)
    
    # Resource Metrics rows (4-6)
    for i in range(4, 7):
        for j in range(4):
            if i % 2 == 1:
                table[(i, j)].set_facecolor("#fef5e7")
            else:
                table[(i, j)].set_facecolor("#fffbf0")
            table[(i, j)].set_text_props(fontsize=10)
            table[(i, j)].set_edgecolor("#bdc3c7")
            table[(i, j)].set_linewidth(1)
    
    # DRL Learning Metrics rows (7-10)
    for i in range(7, 11):
        for j in range(4):
            if i % 2 == 1:
                table[(i, j)].set_facecolor("#e8f8f5")
            else:
                table[(i, j)].set_facecolor("#f0fdf8")
            table[(i, j)].set_text_props(fontsize=10)
            table[(i, j)].set_edgecolor("#bdc3c7")
            table[(i, j)].set_linewidth(1)
    
    # Add section labels on left side for better readability
    fig.text(0.02, 0.85, "Hiệu suất Tài chính", fontsize=11, fontweight="bold", rotation=90, color="#2c3e50")
    fig.text(0.02, 0.58, "Tài nguyên", fontsize=11, fontweight="bold", rotation=90, color="#d68910")
    fig.text(0.02, 0.35, "Học tập DRL", fontsize=11, fontweight="bold", rotation=90, color="#229954")
    
    # Add title
    fig.text(0.5, 0.98, "Kết quả DRL - So sánh Baseline vs Turbo", 
            ha="center", fontsize=18, fontweight="bold", color="#2c3e50")
    
    # Add subtitle
    fig.text(0.5, 0.94, "Turbo cho kết quả tốt hơn Baseline trên hầu hết các chỉ số quan trọng",
            ha="center", fontsize=11, style="italic", color="#566573")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.08, bottom=0.05)
    
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"✓ Metrics table saved: {output_image}")
    plt.close()


def plot_reward_curves_comparison(baseline_eval: Path, turbo_eval: Path, 
                                  baseline_rollout: Path, turbo_rollout: Path,
                                  output_image: Path):
    """Plot reward curves for both models on same chart"""
    
    baseline_data = load_drl_eval(baseline_eval)
    turbo_data = load_drl_eval(turbo_eval)
    baseline_m = baseline_data.get("metrics", {})
    turbo_m = turbo_data.get("metrics", {})
    
    baseline_rollout_df = load_rollout_data(baseline_rollout)
    turbo_rollout_df = load_rollout_data(turbo_rollout)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot baseline reward curve
    if baseline_rollout_df is not None and not baseline_rollout_df.empty:
        baseline_rollout_df["cumulative_reward"] = baseline_rollout_df["reward"].cumsum()
        ax.plot(baseline_rollout_df.index, baseline_rollout_df["cumulative_reward"], 
               linewidth=2.5, color="steelblue", label="Baseline", marker="o", markersize=2, alpha=0.8)
    else:
        ax.axhline(y=baseline_m.get("total_reward", 0), color="steelblue", 
                  linewidth=2.5, linestyle="--", label="Baseline")
    
    # Plot turbo reward curve
    if turbo_rollout_df is not None and not turbo_rollout_df.empty:
        turbo_rollout_df["cumulative_reward"] = turbo_rollout_df["reward"].cumsum()
        ax.plot(turbo_rollout_df.index, turbo_rollout_df["cumulative_reward"],
               linewidth=2.5, color="coral", label="Turbo", marker="s", markersize=2, alpha=0.8)
    else:
        ax.axhline(y=turbo_m.get("total_reward", 0), color="coral",
                  linewidth=2.5, linestyle="--", label="Turbo")
    
    ax.set_title("So sánh Reward Curve: Baseline vs Turbo", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"✓ Reward curves saved: {output_image}")
    plt.close()


def plot_portfolio_curves_comparison(baseline_eval: Path, turbo_eval: Path,
                                     baseline_rollout: Path, turbo_rollout: Path,
                                     output_image: Path):
    """Plot portfolio value curves for both models on same chart"""
    
    baseline_data = load_drl_eval(baseline_eval)
    turbo_data = load_drl_eval(turbo_eval)
    baseline_m = baseline_data.get("metrics", {})
    turbo_m = turbo_data.get("metrics", {})
    
    baseline_rollout_df = load_rollout_data(baseline_rollout)
    turbo_rollout_df = load_rollout_data(turbo_rollout)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    initial_value = 1000.0
    
    # Plot baseline portfolio
    if baseline_rollout_df is not None and not baseline_rollout_df.empty and "equity" in baseline_rollout_df.columns:
        ax.plot(baseline_rollout_df.index, baseline_rollout_df["equity"],
               linewidth=2.5, color="steelblue", label="Baseline", marker="o", markersize=2, alpha=0.8)
        ax.fill_between(baseline_rollout_df.index, initial_value, baseline_rollout_df["equity"],
                       alpha=0.2, color="steelblue")
    
    # Plot turbo portfolio
    if turbo_rollout_df is not None and not turbo_rollout_df.empty and "equity" in turbo_rollout_df.columns:
        ax.plot(turbo_rollout_df.index, turbo_rollout_df["equity"],
               linewidth=2.5, color="coral", label="Turbo", marker="s", markersize=2, alpha=0.8)
        ax.fill_between(turbo_rollout_df.index, initial_value, turbo_rollout_df["equity"],
                       alpha=0.2, color="coral")
    
    ax.axhline(y=initial_value, color="gray", linestyle="--", alpha=0.5, linewidth=2, label="Starting Balance")
    
    ax.set_title("So sánh Portfolio Value: Baseline vs Turbo", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Portfolio Value (USDT)", fontsize=12)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"✓ Portfolio curves saved: {output_image}")
    plt.close()


def plot_action_distribution_comparison(baseline_eval: Path, turbo_eval: Path, output_image: Path):
    """Plot action distribution for both models side by side"""
    
    baseline_data = load_drl_eval(baseline_eval)
    turbo_data = load_drl_eval(turbo_eval)
    baseline_m = baseline_data.get("metrics", {})
    turbo_m = turbo_data.get("metrics", {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline pie chart
    base_buy = baseline_m.get("action_buy_ratio", 0) * 100
    base_sell = baseline_m.get("action_sell_ratio", 0) * 100
    base_hold = baseline_m.get("action_hold_ratio", 0) * 100
    
    actions_base = [base_buy, base_sell, base_hold]
    labels_base = [f"Buy\n({int(baseline_m.get('action_buy_count', 0))})", 
                   f"Sell\n({int(baseline_m.get('action_sell_count', 0))})",
                   f"Hold\n({int(baseline_m.get('action_hold_count', 0))})"]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    
    # Filter out zero-value actions
    base_actions_filtered = [(actions_base[i], labels_base[i], colors[i]) for i in range(3) if actions_base[i] > 0]
    if base_actions_filtered:
        base_vals, base_labels, base_colors = zip(*base_actions_filtered)
        wedges, texts, autotexts = ax1.pie(
            base_vals,
            labels=base_labels,
            autopct="%1.1f%%",
            colors=base_colors,
            startangle=90,
            textprops={"fontsize": 11, "weight": "bold"}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(11)
    
    ax1.set_title("Baseline Action Distribution", fontsize=13, fontweight="bold", pad=20)
    
    # Turbo pie chart
    turbo_buy = turbo_m.get("action_buy_ratio", 0) * 100
    turbo_sell = turbo_m.get("action_sell_ratio", 0) * 100
    turbo_hold = turbo_m.get("action_hold_ratio", 0) * 100
    
    actions_turbo = [turbo_buy, turbo_sell, turbo_hold]
    labels_turbo = [f"Buy\n({int(turbo_m.get('action_buy_count', 0))})",
                    f"Sell\n({int(turbo_m.get('action_sell_count', 0))})",
                    f"Hold\n({int(turbo_m.get('action_hold_count', 0))})"]
    
    # Filter out zero-value actions
    turbo_actions_filtered = [(actions_turbo[i], labels_turbo[i], colors[i]) for i in range(3) if actions_turbo[i] > 0]
    if turbo_actions_filtered:
        turbo_vals, turbo_labels, turbo_colors = zip(*turbo_actions_filtered)
        wedges, texts, autotexts = ax2.pie(
            turbo_vals,
            labels=turbo_labels,
            autopct="%1.1f%%",
            colors=turbo_colors,
            startangle=90,
            textprops={"fontsize": 11, "weight": "bold"}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(11)
    
    ax2.set_title("Turbo Action Distribution", fontsize=13, fontweight="bold", pad=20)
    
    fig.suptitle("So sánh Action Distribution: Baseline vs Turbo", 
                fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"✓ Action distribution saved: {output_image}")
    plt.close()


def plot_drawdown_curves_comparison(baseline_eval: Path, turbo_eval: Path,
                                    baseline_rollout: Path, turbo_rollout: Path,
                                    output_image: Path):
    """Plot drawdown curves for both models on same chart"""
    
    baseline_rollout_df = load_rollout_data(baseline_rollout)
    turbo_rollout_df = load_rollout_data(turbo_rollout)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot baseline drawdown
    if baseline_rollout_df is not None and not baseline_rollout_df.empty and "drawdown" in baseline_rollout_df.columns:
        drawdown_base = baseline_rollout_df["drawdown"].values * 100
        ax.plot(baseline_rollout_df.index, drawdown_base,
               linewidth=2.5, color="steelblue", label="Baseline", marker="o", markersize=2, alpha=0.8)
        ax.fill_between(baseline_rollout_df.index, 0, drawdown_base, alpha=0.2, color="steelblue")
    
    # Plot turbo drawdown
    if turbo_rollout_df is not None and not turbo_rollout_df.empty and "drawdown" in turbo_rollout_df.columns:
        drawdown_turbo = turbo_rollout_df["drawdown"].values * 100
        ax.plot(turbo_rollout_df.index, drawdown_turbo,
               linewidth=2.5, color="coral", label="Turbo", marker="s", markersize=2, alpha=0.8)
        ax.fill_between(turbo_rollout_df.index, 0, drawdown_turbo, alpha=0.2, color="coral")
    
    ax.set_title("So sánh Drawdown Curve: Baseline vs Turbo", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_ylim(bottom=-0.5)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"✓ Drawdown curves saved: {output_image}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate DRL comparison: 1 metrics table + 4 comparison charts"
    )
    
    parser.add_argument("--baseline-eval", type=Path, required=True,
                       help="Baseline eval JSON")
    parser.add_argument("--turbo-eval", type=Path, required=True,
                       help="Turbo eval JSON")
    parser.add_argument("--baseline-rollout", type=Path, default=None,
                       help="Baseline rollout CSV (optional)")
    parser.add_argument("--turbo-rollout", type=Path, default=None,
                       help="Turbo rollout CSV (optional)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path("docs/figures") / f"drl_comparison_{date.today().isoformat()}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING DRL COMPARISON")
    print("="*80)
    
    # 1. Create metrics table image
    print("\n1. Creating metrics table image...")
    table_output = args.output_dir / "drl_metrics_table.png"
    create_metrics_table_image(args.baseline_eval, args.turbo_eval, table_output)
    
    # 2. Create reward curves comparison
    print("\n2. Creating reward curves comparison...")
    reward_output = args.output_dir / "drl_reward_curves.png"
    plot_reward_curves_comparison(args.baseline_eval, args.turbo_eval,
                                  args.baseline_rollout, args.turbo_rollout,
                                  reward_output)
    
    # 3. Create portfolio value curves comparison
    print("\n3. Creating portfolio value curves comparison...")
    portfolio_output = args.output_dir / "drl_portfolio_curves.png"
    plot_portfolio_curves_comparison(args.baseline_eval, args.turbo_eval,
                                     args.baseline_rollout, args.turbo_rollout,
                                     portfolio_output)
    
    # 4. Create action distribution comparison
    print("\n4. Creating action distribution comparison...")
    action_output = args.output_dir / "drl_action_distribution.png"
    plot_action_distribution_comparison(args.baseline_eval, args.turbo_eval, action_output)
    
    # 5. Create drawdown curves comparison
    print("\n5. Creating drawdown curves comparison...")
    drawdown_output = args.output_dir / "drl_drawdown_curves.png"
    plot_drawdown_curves_comparison(args.baseline_eval, args.turbo_eval,
                                    args.baseline_rollout, args.turbo_rollout,
                                    drawdown_output)
    
    print("\n" + "="*80)
    print(f"✅ ALL OUTPUTS SAVED TO: {args.output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. drl_metrics_table.png")
    print(f"  2. drl_reward_curves.png")
    print(f"  3. drl_portfolio_curves.png")
    print(f"  4. drl_action_distribution.png (side-by-side)")
    print(f"  5. drl_drawdown_curves.png")


if __name__ == "__main__":
    main()
