from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class FQ_TurboQuantStrategy(IStrategy):
    """TurboQuant strategy using OHLCV + tq_xhat_* features from freqtrade_dataset.csv.

    Core thesis-oriented logic:
    - Decision is driven primarily by reconstructed features `tq_xhat_*`.
    - Distortion-aware filters are enforced via `tq_error`, `tq_score`, `tq_confidence`.
    """

    INTERFACE_VERSION = 3

    timeframe = "1m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 240

    minimal_roi = {
        "0": 0.015,
        "20": 0.008,
        "60": 0.0,
    }

    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.007
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 8,
        }
    ]

    # Thresholds on reconstructed normalized features (tq_xhat_*)
    tqx_min_trend = -0.15
    tqx_max_volatility = 1.15
    tqx_max_atr = 1.20
    tqx_exit_trend = -0.45

    # Distortion / quality thresholds for thesis claims.
    tq_max_error = 0.35
    tq_exit_error = 0.48
    tq_min_score = -0.05
    tq_exit_score = -0.18
    tq_min_confidence = 0.55
    tq_exit_confidence = 0.42

    plot_config = {
        "main_plot": {
            "tqx_trend": {},
            "tqx_momentum": {},
        },
        "subplots": {
            "TurboQuant": {
                "tq_xhat_rsi": {},
                "tq_xhat_volatility": {},
                "tq_xhat_atr": {},
                "tqx_trend": {},
                "tqx_momentum": {},
            },
            "Quality": {
                "tq_error": {},
                "tq_score": {},
                "tq_confidence": {},
            },
            "OHLCV": {
                "volume_ratio": {},
                "ohlcv_body": {},
            },
        },
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._tqx_features: DataFrame = pd.DataFrame()
        self._tqx_last_mtime_ns: int | None = None

    def _dataset_path(self) -> Path:
        # strategies/ -> user_data/ -> freqtrade_setup/ -> repo root
        return Path(__file__).resolve().parents[3] / "freqtrade_dataset.csv"

    def _reload_tqx_features_if_needed(self) -> None:
        dataset_path = self._dataset_path()
        if not dataset_path.exists():
            self._tqx_features = pd.DataFrame()
            self._tqx_last_mtime_ns = None
            return

        stat = dataset_path.stat()
        if self._tqx_last_mtime_ns == stat.st_mtime_ns and not self._tqx_features.empty:
            return

        required_cols = [
            "time",
            "tq_xhat_log_return",
            "tq_xhat_return_5",
            "tq_xhat_log_volume",
            "tq_xhat_candle_body",
            "tq_xhat_rsi",
            "tq_xhat_macd",
            "tq_xhat_macd_signal",
            "tq_xhat_volatility",
            "tq_xhat_atr",
            "tq_error",
            "tq_score",
            "tq_confidence",
        ]

        tqx_df = pd.read_csv(dataset_path, usecols=required_cols)
        tqx_df["date"] = pd.to_datetime(tqx_df["time"], errors="coerce")
        tqx_df = tqx_df.dropna(subset=["date"])

        for col in required_cols[1:]:
            tqx_df[col] = pd.to_numeric(tqx_df[col], errors="coerce")

        tqx_df = tqx_df[["date"] + required_cols[1:]]
        tqx_df = tqx_df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        self._tqx_features = tqx_df.set_index("date")
        self._tqx_last_mtime_ns = stat.st_mtime_ns

    def _merge_tqx_features(self, dataframe: DataFrame) -> DataFrame:
        self._reload_tqx_features_if_needed()

        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")

        tqx_cols = [
            "tq_xhat_log_return",
            "tq_xhat_return_5",
            "tq_xhat_log_volume",
            "tq_xhat_candle_body",
            "tq_xhat_rsi",
            "tq_xhat_macd",
            "tq_xhat_macd_signal",
            "tq_xhat_volatility",
            "tq_xhat_atr",
            "tq_error",
            "tq_score",
            "tq_confidence",
        ]

        if self._tqx_features.empty:
            for col in tqx_cols:
                dataframe[col] = 0.0
            dataframe["tq_error"] = 999.0
            return dataframe

        tqx_aligned = self._tqx_features.reindex(pd.DatetimeIndex(dataframe["date"]))
        for col in tqx_cols:
            dataframe[col] = tqx_aligned[col].to_numpy()

        for col in tqx_cols:
            if col == "tq_error":
                dataframe[col] = dataframe[col].fillna(999.0)
            else:
                dataframe[col] = dataframe[col].fillna(0.0)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dataframe.copy()

        dataframe["volume_mean"] = dataframe["volume"].rolling(20, min_periods=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_mean"].replace(0, np.nan)
        dataframe["ohlcv_body"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"].replace(0, np.nan)

        dataframe = self._merge_tqx_features(dataframe)

        dataframe["tqx_trend"] = dataframe["tq_xhat_macd"] - dataframe["tq_xhat_macd_signal"]
        dataframe["tqx_momentum"] = dataframe["tq_xhat_log_return"] + 0.35 * dataframe["tq_xhat_return_5"]

        numeric_cols = [
            "volume_mean",
            "volume_ratio",
            "ohlcv_body",
            "tq_xhat_log_return",
            "tq_xhat_return_5",
            "tq_xhat_rsi",
            "tq_xhat_macd",
            "tq_xhat_macd_signal",
            "tq_xhat_volatility",
            "tq_xhat_atr",
            "tq_error",
            "tq_score",
            "tq_confidence",
            "tqx_trend",
            "tqx_momentum",
        ]

        dataframe[numeric_cols] = dataframe[numeric_cols].replace([np.inf, -np.inf], np.nan)
        dataframe["tq_error"] = dataframe["tq_error"].fillna(999.0)
        dataframe[[c for c in numeric_cols if c != "tq_error"]] = dataframe[
            [c for c in numeric_cols if c != "tq_error"]
        ].fillna(0.0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > 0,
            dataframe["volume_ratio"] > 0.95,
            dataframe["tq_xhat_rsi"].between(-0.9, 0.9),
            dataframe["tqx_trend"] >= self.tqx_min_trend,
            dataframe["tqx_momentum"] > -0.2,
            dataframe["tq_xhat_volatility"] <= self.tqx_max_volatility,
            dataframe["tq_xhat_atr"] <= self.tqx_max_atr,
            dataframe["tq_error"] <= self.tq_max_error,
            dataframe["tq_score"] >= self.tq_min_score,
            dataframe["tq_confidence"] >= self.tq_min_confidence,
            dataframe["ohlcv_body"] > -0.02,
        ]

        dataframe.loc[
            pd.concat(conditions, axis=1).all(axis=1),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > 0,
            (
                (dataframe["tqx_trend"] <= self.tqx_exit_trend)
                | (dataframe["tqx_momentum"] < -0.55)
                | (dataframe["tq_xhat_volatility"] > 1.25)
                | (dataframe["tq_xhat_atr"] > 1.30)
                | (dataframe["tq_error"] >= self.tq_exit_error)
                | (dataframe["tq_score"] <= self.tq_exit_score)
                | (dataframe["tq_confidence"] < self.tq_exit_confidence)
            ),
        ]

        dataframe.loc[
            pd.concat(conditions, axis=1).all(axis=1),
            "exit_long",
        ] = 1

        return dataframe