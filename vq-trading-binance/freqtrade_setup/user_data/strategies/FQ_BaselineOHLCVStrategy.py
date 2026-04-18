from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class FQ_BaselineOHLCVStrategy(IStrategy):
    """Baseline Freqtrade strategy using only OHLCV-derived indicators.

    This version is intentionally independent from TurboQuant/VQ so it can be
    used as the clean benchmark in backtests.
    """

    INTERFACE_VERSION = 3

    timeframe = "1m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 240

    # Lịch ROI theo thời gian giữ lệnh
    minimal_roi = {
        "0": 0.012,
        "20": 0.006,
        "60": 0.0,
    }

    # Cắt lỗ tối đa 3%
    stoploss = -0.03

    # Cấu hình trailing stop để bảo vệ lợi nhuận sẽ kích hoạt khi đạt lợi nhuận
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 5,
        }
    ]

    plot_config = {
        "main_plot": {
            "n_macd": {},
            "n_macd_signal": {},
        },
        "subplots": {
            "Normalized": {
                "n_rsi": {},
                "n_return_5": {},
                "n_volatility": {},
                "n_atr": {},
                "n_trend": {},
            },
            "OHLCV": {
                "ohlcv_spread": {},
                "ohlcv_body": {},
            },
        },
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._n_features: DataFrame = pd.DataFrame()
        self._n_last_mtime_ns: int | None = None

    def _dataset_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "freqtrade_dataset.csv"

    def _reload_n_features_if_needed(self) -> None:
        dataset_path = self._dataset_path()
        if not dataset_path.exists():
            self._n_features = pd.DataFrame()
            self._n_last_mtime_ns = None
            return

        stat = dataset_path.stat()
        if self._n_last_mtime_ns == stat.st_mtime_ns and not self._n_features.empty:
            return

        required_cols = [
            "time",
            "n_log_return",
            "n_return_5",
            "n_log_volume",
            "n_candle_body",
            "n_rsi",
            "n_macd",
            "n_macd_signal",
            "n_volatility",
            "n_atr",
        ]

        n_df = pd.read_csv(dataset_path, usecols=required_cols)
        n_df["date"] = pd.to_datetime(n_df["time"], errors="coerce")
        n_df = n_df.dropna(subset=["date"])

        for col in required_cols[1:]:
            n_df[col] = pd.to_numeric(n_df[col], errors="coerce")

        n_df = n_df[["date"] + required_cols[1:]]
        n_df = n_df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        self._n_features = n_df.set_index("date")
        self._n_last_mtime_ns = stat.st_mtime_ns

    def _merge_n_features(self, dataframe: DataFrame) -> DataFrame:
        self._reload_n_features_if_needed()

        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")

        n_cols = [
            "n_log_return",
            "n_return_5",
            "n_log_volume",
            "n_candle_body",
            "n_rsi",
            "n_macd",
            "n_macd_signal",
            "n_volatility",
            "n_atr",
        ]

        if self._n_features.empty:
            for col in n_cols:
                dataframe[col] = 0.0
            return dataframe

        n_aligned = self._n_features.reindex(pd.DatetimeIndex(dataframe["date"]))
        for col in n_cols:
            dataframe[col] = n_aligned[col].to_numpy()
            dataframe[col] = dataframe[col].fillna(0.0)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dataframe.copy()

        dataframe = self._merge_n_features(dataframe)

        dataframe["n_trend"] = dataframe["n_macd"] - dataframe["n_macd_signal"]
        dataframe["ohlcv_spread"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"].replace(0, np.nan)
        dataframe["ohlcv_body"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"].replace(0, np.nan)
        dataframe["ohlcv_green"] = (dataframe["close"] > dataframe["open"]).astype(int)

        dataframe[[
            "n_log_return",
            "n_return_5",
            "n_log_volume",
            "n_candle_body",
            "n_rsi",
            "n_macd",
            "n_macd_signal",
            "n_volatility",
            "n_atr",
            "n_trend",
            "ohlcv_spread",
            "ohlcv_body",
        ]] = dataframe[[
            "n_log_return",
            "n_return_5",
            "n_log_volume",
            "n_candle_body",
            "n_rsi",
            "n_macd",
            "n_macd_signal",
            "n_volatility",
            "n_atr",
            "n_trend",
            "ohlcv_spread",
            "ohlcv_body",
        ]].replace([float("inf"), float("-inf")], np.nan)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > 0,
            dataframe["n_rsi"].between(-0.75, 0.75),
            dataframe["n_macd"] > dataframe["n_macd_signal"],
            dataframe["n_trend"] > -0.05,
            dataframe["n_return_5"] > -0.25,
            dataframe["n_volatility"] < 1.0,
            dataframe["n_atr"] < 1.20,
            dataframe["n_log_return"] > -0.70,
            dataframe["ohlcv_green"] == 1,
            dataframe["ohlcv_spread"].between(0.00005, 0.020),
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
                (dataframe["n_macd"] < dataframe["n_macd_signal"])
                | (dataframe["n_trend"] < -0.40)
                | (dataframe["n_return_5"] < -0.5)
                | (dataframe["n_volatility"] > 1.25)
                | (dataframe["n_atr"] > 1.30)
                | (dataframe["n_rsi"] < -1.00)
                | (dataframe["n_rsi"] > 1.10)
            ),
        ]

        dataframe.loc[
            pd.concat(conditions, axis=1).all(axis=1),
            "exit_long",
        ] = 1

        return dataframe
