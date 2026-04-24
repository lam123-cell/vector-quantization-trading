"""
STEP 7 — TRAIN DRL AGENT

Train PPO agent on baseline or turbo feature set.
PPO = Proximal Policy Optimization (thuật toán từ stable-baselines3)

Cách dùng:
    # Train baseline
    python scripts/train_drl.py --feature-set baseline --timesteps 100000 --output models/ppo_baseline
    
    # Train turbo
    python scripts/train_drl.py --feature-set turbo --timesteps 100000 --output models/ppo_turbo
    
    # Google Colab: Copy lệnh này vào cell
    %cd /content/vector-quantization-trading/vq-trading-binance
    !python scripts/train_drl.py --feature-set baseline --timesteps 50000 --output models/ppo_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.drl_env import TradingEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]
    
    parser = argparse.ArgumentParser(
        description="Train PPO agent on DRL trading environment"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["baseline", "turbo"],
        default="baseline",
        help="Feature set to use (baseline=n_*, turbo=tq_xhat_*)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to training dataset (auto-detected if not specified)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps before updating policy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "models" / "ppo_baseline",
        help="Output directory for model checkpoint",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=info, 2=debug)",
    )
    
    return parser.parse_args()


def resolve_dataset_path(feature_set: Literal["baseline", "turbo"], explicit: Path | None) -> Path:
    """Resolve training dataset path."""
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"Dataset not found: {explicit}")
        return explicit
    
    repo_root = Path(__file__).resolve().parents[1]
    
    if feature_set == "baseline":
        dataset_path = repo_root / "drl_dataset_baseline_train.csv"
    else:
        dataset_path = repo_root / "drl_dataset_turbo_train.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run: python scripts/build_dataset.py first"
        )
    
    return dataset_path


def create_environment(dataset_path: Path) -> TradingEnv:
    """Create trading environment from dataset."""
    print(f"[*] Loading dataset: {dataset_path.name}")
    data = pd.read_csv(dataset_path)
    print(f"    - Rows: {len(data):,}")
    print(f"    - Columns: {len(data.columns)}")
    
    # Auto-detect feature columns
    feature_columns = sorted([col for col in data.columns if col.startswith("n_") or col.startswith("tq_xhat_")])
    if not feature_columns:
        raise ValueError("No feature columns found in dataset")
    
    print(f"    - Feature columns: {len(feature_columns)}")
    print(f"      {', '.join(feature_columns[:5])}{'...' if len(feature_columns) > 5 else ''}")
    
    env = TradingEnv(
        data=data,
        feature_columns=feature_columns,
        initial_balance=1000.0,
        transaction_cost=0.001,  # 0.1%
    )
    
    return env


def train_ppo(
    env: TradingEnv,
    timesteps: int,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    output_dir: Path,
    verbose: int,
) -> PPO:
    """Train PPO model on environment."""
    print(f"\n[*] Creating PPO model")
    print(f"    - Learning rate: {learning_rate}")
    print(f"    - Batch size: {batch_size}")
    print(f"    - N-steps: {n_steps}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        verbose=verbose,
        tensorboard_log=str(output_dir / "logs"),
    )
    
    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, timesteps // 10),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo",
        save_replay_buffer=False,
    )
    
    print(f"\n[*] Training PPO for {timesteps:,} timesteps")
    print(f"    - Output: {output_dir}")
    print("=" * 80)
    
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    print("=" * 80)
    print(f"[✓] Training complete!")
    
    return model


def save_model(model: PPO, env: TradingEnv, output_dir: Path) -> None:
    """Save trained model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "ppo_model"
    model.save(str(model_path))
    print(f"[*] Saved model: {model_path}.zip")
    
    # Save environment info
    info = {
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "n_features": env.n_features,
        "feature_columns": env.feature_columns,
        "initial_balance": env.initial_balance,
        "transaction_cost": env.transaction_cost,
    }
    
    import json
    info_path = output_dir / "env_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"[*] Saved environment info: {info_path}")
    
    # Save training config
    config = {
        "timesteps": model.num_timesteps,
        "policy": "MlpPolicy",
        "learning_rate": float(model.learning_rate),
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[*] Saved training config: {config_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    print("=" * 80)
    print("STEP 7 — TRAIN DRL AGENT (PPO)")
    print("=" * 80)
    print(f"\n[Configuration]")
    print(f"  Feature set: {args.feature_set}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Output: {args.output}")
    print()
    
    try:
        # Resolve dataset path
        dataset_path = resolve_dataset_path(args.feature_set, args.dataset)
        
        # Create environment
        env = create_environment(dataset_path)
        print(f"\n[✓] Environment created!")
        print(f"    - Observation space: {env.observation_space}")
        print(f"    - Action space: {env.action_space}")
        
        # Train model
        model = train_ppo(
            env=env,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            output_dir=args.output,
            verbose=args.verbose,
        )
        
        # Save model
        save_model(model, env, args.output)
        
        print(f"\n[✓] Training complete!")
        print(f"    Model saved to: {args.output}")
        print(f"\n[Next steps]")
        print(f"  1. Evaluate on test set:")
        print(f"     python scripts/evaluate_drl.py --model {args.output}/ppo_model.zip --feature-set {args.feature_set}")
        print(f"  2. Transfer to Google Colab for faster training")
        
        return 0
        
    except Exception as e:
        print(f"\n[✗] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
