"""
DRL Trading Environment for Vector Quantization Comparison

Supports both baseline (n_*) and turbo (tq_xhat_*) feature sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


FeatureSetName = Literal["baseline", "turbo"]


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for DRL agents.
    
    Action Space:
        - 0: HOLD
        - 1: BUY
        - 2: SELL
    
    Observation Space:
        - Box(low=-inf, high=inf, shape=(n_features,)) - state vector for current step
    
    Attributes:
        data: DataFrame with columns [time, open, high, low, close, volume, n_*/tq_xhat_*]
        feature_columns: List of feature names used for state representation
        current_step: Current index in the data
    """

    metadata = {"render_modes": ["human"]}

    # Action space constants
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        initial_balance: float = 1000.0,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize trading environment.

        Args:
            data: DataFrame with OHLCV + features (time, open, high, low, close, volume, features...)
            feature_columns: Names of feature columns to use as state vector
            initial_balance: Starting account balance
            transaction_cost: Transaction cost as fraction (0.1% = 0.001)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        super().__init__()

        # Validate inputs
        if data.empty:
            raise ValueError("data DataFrame must not be empty")
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")

        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns in data: {missing_columns}")

        # Store data and configuration
        self.data = data.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        # ========================
        # ACTION SPACE
        # ========================
        # 3 actions: Hold (0), Buy (1), Sell (2)
        self.action_space = spaces.Discrete(3)

        # ========================
        # OBSERVATION SPACE
        # ========================
        # State = feature vector [n_rsi, n_macd, ..., ] or [tq_xhat_rsi, tq_xhat_macd, ...]
        # Features are typically normalized to [-1, 1] or [-3, 3], so use inf bounds for safety
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # ========================
        # TRADING STATE
        # ========================
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # BTC held
        self.entry_price = 0.0  # Buy-in price
        self.trades = []  # History of trades
        self.previous_price = 0.0  # For computing price change reward

        # ========================
        # METRICS
        # ========================
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.num_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.cumulative_reward = 0.0

    # ========================
    # GYMNASIUM API
    # ========================

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns:
            observation: Initial state vector (feature values at step 0)
            info: Metadata dictionary
        """
        super().reset(seed=seed)

        # Reset to step 0
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.previous_price = 0.0
        self.trades = []

        self.total_profit = 0.0
        self.total_loss = 0.0
        self.num_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.cumulative_reward = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step of the environment.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            observation: New state vector
            reward: Scalar reward
            terminated: Episode end flag
            truncated: Episode truncation flag
            info: Metadata
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Get current and previous prices
        current_row = self.data.iloc[self.current_step]
        current_price = float(current_row["close"])

        # Initialize previous price on first step
        if self.current_step == 0:
            self.previous_price = current_price

        # Execute action and remember whether an order was filled.
        traded = self._execute_action(action, current_price)

        # Calculate reward for this step
        reward = self._calculate_reward(action, current_price, traded)
        self.cumulative_reward += reward

        # Update previous price for next step
        self.previous_price = current_price

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.n_features, dtype=np.float32)
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render environment state."""
        if self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]
            print(f"[Step {self.current_step}]")
            print(f"  Time: {row['time']}")
            print(f"  Price: {row['close']:.2f}")
            print(f"  Balance: {self.balance:.2f}")
            print(f"  Position: {self.position:.4f} BTC")

    # ========================
    # STEP 4 - STATE CONSTRUCTION
    # ========================

    def _get_observation(self) -> np.ndarray:
        """
        Extract current state from feature columns.

        State = feature vector at current step:
        - Baseline: [n_log_return, n_rsi, n_macd, n_volatility, n_atr, ...]
        - Turbo:    [tq_xhat_log_return, tq_xhat_rsi, tq_xhat_macd, tq_xhat_volatility, tq_xhat_atr, ...]

        Returns:
            np.ndarray: Feature vector of shape (n_features,)
        """
        if self.current_step >= len(self.data):
            return np.zeros(self.n_features, dtype=np.float32)

        row = self.data.iloc[self.current_step]
        state = np.array(
            [float(row[col]) for col in self.feature_columns],
            dtype=np.float32,
        )

        return state

    def _get_info(self) -> dict[str, Any]:
        """Return metadata about current step."""
        if self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]
            return {
                "time": row.get("time", "N/A"),
                "step": self.current_step,
                "balance": self.balance,
                "position": self.position,
                "price": float(row["close"]),
            }
        return {"step": self.current_step}

    # ========================
    # STEP 5 - REWARD FUNCTION
    # ========================

    def _calculate_reward(self, action: int, current_price: float, traded: bool) -> float:
        """
        Calculate reward for current step.

        Formula:
            r_t = (P_t - P_{t-1}) * position - transaction_cost - risk_penalty

        Components:
            - Price change reward: (current_price - previous_price) * position
              - If holding (position > 0) and price goes up, reward is positive
              - If holding and price goes down, reward is negative
            
            - Transaction cost: Penalty for trading (BUY/SELL actions)
              - Applies when action changes position
            
            - Risk penalty: Additional cost for holding risk overnight
              - Proportional to position held
              - Encourages agent to close positions

        Args:
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            current_price: Current asset price
            traded: Whether an order was executed at this step

        Returns:
            float: Reward value for this step
        """
        # 1. Price change reward from holding position
        price_change = current_price - self.previous_price
        holding_reward = price_change * self.position

        # 2. Transaction cost penalty when trading
        transaction_cost_penalty = 0.0
        if traded and action in (self.BUY, self.SELL):
            # Cost of filled order (commission applied during execution)
            transaction_cost_penalty = -self.transaction_cost * current_price

        # 3. Risk penalty: Cost of holding position overnight
        # Discourages agent from holding too long without clear reason
        # Calibrated based on position size and implied volatility
        risk_penalty = 0.0
        if self.position > 0.0:
            # Get volatility feature if available for risk assessment
            volatility = self._get_volatility_estimate()
            # Base risk cost + scaled by volatility
            risk_penalty = -(0.0001 + 0.0001 * abs(volatility)) * current_price

        # 4. Combine all reward components
        reward = holding_reward + transaction_cost_penalty + risk_penalty

        return float(reward)

    def _get_volatility_estimate(self) -> float:
        """
        Extract volatility estimate from features if available.

        Returns:
            float: Volatility value (or 0.0 if not available)
        """
        if self.current_step >= len(self.data):
            return 0.0

        row = self.data.iloc[self.current_step]

        # Try to find volatility feature
        for col_name in ["volatility", "n_volatility", "tq_xhat_volatility"]:
            if col_name in row:
                try:
                    return float(row[col_name])
                except (ValueError, TypeError):
                    pass

        return 0.0

    # ========================
    # STEP 3 - ACTION EXECUTION
    # ========================

    def _execute_action(self, action: int, current_price: float) -> bool:
        """
        Execute action and update account state.

        Actions:
            0 = HOLD: Do nothing
            1 = BUY: Buy with available cash (fractional quantity allowed)
            2 = SELL: Close full position if open

        Args:
            action: Action index
            current_price: Current asset price

        Returns:
            bool: True if a trade was executed, otherwise False
        """
        if action == self.HOLD:
            # Do nothing
            return False

        elif action == self.BUY:
            # Open position using available balance (supports small accounts on high-priced assets).
            if self.position == 0.0 and self.balance > 0.0:
                qty = self.balance / (current_price * (1 + self.transaction_cost))
                if qty > 0.0:
                    cost = qty * current_price * (1 + self.transaction_cost)
                    self.position = float(qty)
                    self.entry_price = current_price
                    self.balance -= cost
                    self.num_trades += 1
                    self.trades.append(
                        {
                            "step": self.current_step,
                            "type": "BUY",
                            "price": current_price,
                            "balance_before": self.balance + cost,
                            "quantity": self.position,
                        }
                    )
                    return True
            return False

        elif action == self.SELL:
            # Close full position if open
            if self.position > 0.0:
                position_size = self.position
                proceeds = position_size * current_price * (1 - self.transaction_cost)
                buy_cost = position_size * self.entry_price * (1 + self.transaction_cost)
                pnl = proceeds - buy_cost
                self.balance += proceeds
                self.position = 0.0

                if pnl > 0:
                    self.total_profit += pnl
                    self.win_trades += 1
                else:
                    self.total_loss += abs(pnl)
                    self.loss_trades += 1

                self.num_trades += 1
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "SELL",
                        "price": current_price,
                        "pnl": pnl,
                            "quantity": position_size,
                        "balance_after": self.balance,
                    }
                )
                return True
            return False

        return False

    # ========================
    # METRICS & PROPERTIES
    # ========================

    def get_total_value(self) -> float:
        """Calculate total portfolio value (balance + position value)."""
        if self.current_step < len(self.data):
            current_price = float(self.data.iloc[self.current_step]["close"])
            return self.balance + self.position * current_price
        return self.balance

    def get_portfolio_return(self) -> float:
        """Calculate simple return from initial balance."""
        total = self.get_total_value()
        if self.initial_balance > 0:
            return (total - self.initial_balance) / self.initial_balance
        return 0.0

    def get_metrics(self) -> dict[str, float]:
        """Return summary metrics for evaluation."""
        total_value = self.get_total_value()
        total_return = self.get_portfolio_return()
        win_rate = self.win_trades / self.num_trades if self.num_trades > 0 else 0.0
        avg_profit = self.total_profit / self.win_trades if self.win_trades > 0 else 0.0

        return {
            "total_value": total_value,
            "total_return": total_return,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "num_trades": self.num_trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
        }

    # ========================
    # FACTORY METHODS
    # ========================

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        feature_set: FeatureSetName = "baseline",
        **kwargs: Any,
    ) -> TradingEnv:
        """
        Create environment from CSV file.

        Args:
            csv_path: Path to drl_dataset_baseline.csv or drl_dataset_turbo.csv
            feature_set: "baseline" or "turbo"
            **kwargs: Additional arguments to pass to __init__

        Returns:
            TradingEnv instance
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        data = pd.read_csv(csv_path)

        # Auto-detect feature columns
        if feature_set == "baseline":
            feature_columns = sorted([col for col in data.columns if col.startswith("n_")])
        elif feature_set == "turbo":
            feature_columns = sorted([col for col in data.columns if col.startswith("tq_xhat_")])
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")

        if not feature_columns:
            raise ValueError(f"No feature columns found for {feature_set}")

        return cls(data=data, feature_columns=feature_columns, **kwargs)

    @classmethod
    def from_baseline_csv(cls, csv_path: str | Path, **kwargs: Any) -> TradingEnv:
        """Create environment from baseline dataset."""
        return cls.from_csv(csv_path, feature_set="baseline", **kwargs)

    @classmethod
    def from_turbo_csv(cls, csv_path: str | Path, **kwargs: Any) -> TradingEnv:
        """Create environment from turbo dataset."""
        return cls.from_csv(csv_path, feature_set="turbo", **kwargs)


if __name__ == "__main__":
    # Example usage
    import sys

    print("=" * 80)
    print("DRL Trading Environment - Baseline Example")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[2]
    baseline_path = repo_root / "drl_dataset_baseline.csv"

    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found")
        print("Please run: python scripts/build_dataset.py first")
        sys.exit(1)

    # Create environment
    env = TradingEnv.from_baseline_csv(baseline_path)
    print(f"\n[Environment Created]")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Data length: {len(env.data)}")
    print(f"  Feature columns: {env.feature_columns}")

    # Reset and get initial state
    obs, info = env.reset()
    print(f"\n[Initial State]")
    print(f"  State shape: {obs.shape}")
    print(f"  State values: {obs[:5]}...")  # Print first 5 features
    print(f"  Info: {info}")

    # Test a few steps
    print(f"\n[Testing 3 steps]")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = ["HOLD", "BUY", "SELL"][action]
        print(f"  Step {i + 1}: action={action_name}, reward={reward:.4f}, price={info.get('price', 'N/A')}")

    print(f"\n[Metrics]")
    metrics = env.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n" + "=" * 80)
