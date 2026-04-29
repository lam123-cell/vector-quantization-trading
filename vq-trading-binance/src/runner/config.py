from pathlib import Path

class Config:
    # =========================
    # DATA
    # =========================
    SYMBOL = "btcusdt"
    TIMEFRAME = "1m"
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = str(BASE_DIR / "datasets" / "dataset_master.csv")
    DATA_PATH = str(BASE_DIR / "btc_buffer.csv")

    # =========================
    # FEATURE
    # =========================
    FEATURE_DIM = 9

    # =========================
    # TURBO QUANT
    # =========================
    USE_TURBO = True
    TQ_LEVELS = 16
    TQ_RANGE = (-3, 3)

    # =========================
    # MODEL
    # =========================
    MODEL_TYPE = "baseline"
    # options:
    # "baseline"
    # "tq"
    # "lstm"
    # "drl"
    # "freqtrade"

    # =========================
    # BUFFER
    # =========================
    BUFFER_SIZE = 200
    MIN_DATA = 50
    
    # cứ 20 nến save 1 lần
    SAVE_INTERVAL = 1