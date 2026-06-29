# Vector Quantization for Streaming Financial Data and Trading Models

> Applying **TurboQuant (Online Vector Quantization)** to compress streaming financial data while preserving important information for machine learning-based trading systems.

---

## Overview

This project investigates the application of **Vector Quantization (VQ)** in **online/streaming financial data**.

Modern trading systems continuously receive enormous amounts of market data. Processing every feature vector with full precision requires significant computational resources and memory.

This project explores whether **TurboQuant**, an online vector quantization technique, can compress streaming feature vectors while maintaining enough information for downstream machine learning models.

The compressed data is evaluated using three different approaches:

- Deep Reinforcement Learning (PPO)
- Long Short-Term Memory (LSTM)
- Freqtrade algorithmic trading framework

Finally, the project compares trading performance between:

- **Baseline** (Original features)
- **TurboQuant** (Compressed features)

---

# Motivation

Cryptocurrency exchanges such as Binance generate a continuous stream of market data.

Each timestamp contains dozens of numerical features:

- Open
- High
- Low
- Close
- Volume
- RSI
- MACD
- ATR
- Volatility
- ...

For long-running trading systems, storing and processing these high-dimensional vectors becomes increasingly expensive.

The goal of this research is to answer an important question:

> Can we compress streaming financial feature vectors while preserving enough information for machine learning models?

---

# Project Objectives

The project aims to:

- Study Vector Quantization for streaming data
- Apply TurboQuant to compress financial feature vectors
- Evaluate reconstruction quality after compression
- Compare ML performance using original and compressed data
- Measure the trade-off between compression and trading performance

---

# Project Pipeline

```
               Binance BTC/USDT
                      │
                      ▼
             Historical OHLCV Data
                      │
                      ▼
             Feature Engineering
        (Price, Volume, RSI, MACD ...)
                      │
          ┌───────────┴────────────┐
          │                        │
          ▼                        ▼
     Baseline Features      TurboQuant
       (Original)        (Compressed Features)
          │                        │
          ├────────────┬───────────┤
          ▼            ▼           ▼
         LSTM         DRL      Freqtrade
          │            │           │
          └────────────┴───────────┘
                      │
                      ▼
         Performance Comparison
```

---

# Dataset

The project uses historical cryptocurrency data from **Binance**.

**Trading Pair**

```
BTC/USDT
```

**Primary Timeframe**

```
1 minute
```

Input features include:

- OHLCV
- RSI
- MACD
- ATR
- Volatility
- Engineered financial indicators

---

# TurboQuant

Instead of feeding original vectors directly into machine learning models,

TurboQuant compresses each feature vector while minimizing distortion.

The objective is:

- reduce memory usage
- reduce computational cost
- preserve important market information

The project mainly uses the **MSE-optimized TurboQuant** variant.

---

# Machine Learning Models

## 1. Deep Reinforcement Learning

Algorithm:

```
PPO
```

State

```
Compressed / Original Features
```

Actions

```
Buy
Sell
Hold
```

Reward

```
Reward = Profit − Risk
```

---

## 2. LSTM

Sequence model for financial time-series prediction.

The LSTM predicts trading actions based on sequential market information.

---

## 3. Freqtrade

A rule-based algorithmic trading framework used for:

- Backtesting
- Strategy evaluation
- Risk analysis
- Trading performance comparison

---

# Experiments

Two experimental pipelines are built.

## Baseline

Original feature vectors

↓

Machine Learning Models

↓

Evaluation

---

## TurboQuant

Compressed feature vectors

↓

Machine Learning Models

↓

Evaluation

---

The project compares whether feature compression affects trading performance.

---

# Evaluation Metrics

## Compression Metrics

- MSE
- Reconstruction Error
- Distortion

---

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score

---

## Trading Metrics

- Total Profit
- Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Portfolio Value
- Cumulative Reward

---

# Project Structure

```
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── turboquant/
│
├── src/
│   ├── features/
│   ├── turboquant/
│   ├── drl/
│   ├── lstm/
│   ├── freqtrade/
│   └── evaluate/
│
├── scripts/
│
├── models/
│
├── experiments/
│
├── docs/
│   └── figures/
│
├── freqtrade_setup/
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your_username/vector-quantization-trading.git
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```powershell
.\.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Data Preparation

Generate datasets

```bash
python scripts/build_dataset.py --mode both
```

---

# Train Deep Reinforcement Learning

Baseline

```bash
python scripts/train_drl.py \
--feature-set baseline \
--timesteps 100000 \
--output models/ppo_baseline
```

TurboQuant

```bash
python scripts/train_drl.py \
--feature-set turbo \
--timesteps 100000 \
--output models/ppo_turbo
```

---

# Evaluate DRL

```bash
python scripts/evaluate_drl.py
```

---

# Evaluate LSTM

```bash
python src/evaluate/eval_lstm_baseline.py

python src/evaluate/eval_lstm_tq.py
```

---

# Run Freqtrade

```bash
cd freqtrade_setup
```

Baseline

```bash
docker compose run --rm freqtrade backtesting \
--strategy FQ_BaselineFairStrategy
```

TurboQuant

```bash
docker compose run --rm freqtrade backtesting \
--strategy FQ_TurboCoreFairStrategy
```

---

# Results

The project evaluates:

- Data distortion after compression
- LSTM prediction performance
- PPO trading performance
- Freqtrade backtesting performance

The comparison focuses on whether TurboQuant preserves sufficient information for downstream trading models.

---

# Future Work

Possible extensions include:

- Real-time streaming deployment
- Live Binance API integration
- Transformer-based forecasting
- SAC / DDPG reinforcement learning
- Multi-asset trading
- Adaptive online quantization

---

# Technologies

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- Pandas
- NumPy
- TA-Lib
- Freqtrade
- Docker

---

# Author

Graduation Thesis

**Applying Vector Quantization to Streaming Financial Data for Automated Trading Systems**

University of Transport Ho Chi Minh City

2026
