# BTC Stacking Ensemble Trading System

This project utilized a machine learning model in a pipeline for a Bitcoin (BTC) quantitative trading system, aiming to assist in BTC portfolio allocation. The system ingests multi-source market data: On-chain metrics, Derivatives/Funding Rates, and Technical Spot Price indicators, and trains a domain-segmented meta-learner (using time-series splitting). The predicted probability is evaluated together with RSI momentum filters and dynamic Average True Range (ATR) trailing stops in making a decision on positioning between cash and BTC in the portfolio. 

---

## 1. Summary & Strategy Performance

The backtest on out-of-fold predictions from **July 2023 to July 2026** with ATR Multiplier: 1.0 and RSI Gate: 75.0 (the stabilized setting) yields the following results compared to the passive BTC Buy & Hold Benchmark: 

## 📊 Strategy Performance vs. BTC Benchmark

| Metric | Strategy (Optimal Plateau) | BTC Buy & Hold | Institutional Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Net Return** | **+137.36%** | +110.53% | > BTC Benchmark | 🟢 Pass |
| **Max Drawdown** | **-8.31%** | -53.06% | > -20.00% | 🟢 Pass |
| **Sharpe Ratio** | **2.01** | 0.74 | > 1.20 | 🟢 Pass |
| **Sortino Ratio** | **3.21** | 1.11 | > 1.50 | 🟢 Pass |
| **Calmar Ratio** | **3.80** | 0.50 | > 2.00 | 🟢 Pass |
| **Annualized Alpha** | **+27.16%** | 0.00% | > 0.00% | 🟢 Pass |
| **Beta vs BTC** | **0.17** | 1.00 | < 0.70 | 🟢 Pass |
| **Win Rate** | **47.38%** | 49.96% | 35.0% - 55.0% | 🟢 Pass |
| **Profit Factor** | **1.70** | 1.12 | > 1.50 | 🟢 Pass |

> **Key Takeaway:** The quantitative framework achieved a **+26.83% net return outperformance** while reducing peak-to-trough drawdown from **-53.06% down to -8.31%**, achieving **Sortino Ratio of 3.21** and **Calmar Ratio of 3.80**.

---

## 2. Repository Architecture

```text
btc_stacking_system/
├── data/
│   ├── raw/                        # Ingested datasets (BGeometrics, Binance, Yahoo Finance)
│   ├── processed/                  # Master feature matrix, holdout predictions, Equity Curve of backtesting
│   └── paper_trading.db		  	 # SQLite database tracking forward paper trades
├── models/
│   └── stacking_ensemble.pkl       # Serialized trained stacking meta-learner
├── src/
│   ├── __init__.py
│   ├── config.py                   # Feature domain definitions & hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── download.py             # Multi-source data downloader
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py       # Featuring the downloaded data and setting the target
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_pipelines.py       # Base-learners; On-chain, Derivatives and Technical Indicators 
│   │   ├── meta_learner.py         # DomainStackingClassifier implementation
│   │   └── train_model.py          # Generate OOF prediction via time-series splitting 
│   └── backtest/
│       ├── __init__.py
│       └── engine.py               # Backtesting with Vectorized simulator (Dynamic Sizing + ATR/RSI Gates)
├── .github/workflows/
│   └── paper_trade.yml             # Automated daily execution pipeline
├── app.py			                 # Interactive dashboard
├── paper_trade.py				  	 # Forward testing script 
├── requirements.txt                # Project dependencies
└── README.md                       # System documentation
```

---

## 3. Machine Learning Pipeline

### Data Ingestion & Feature Domains

The pipeline aggregates three data streams and puts each in its own base pipeline:

1. **On-Chain Domain (`XGBoost`):** BGeometrics metrics such as AVIV, MVRV-Z, NUPL, SOPR, and Age cohort. 
2. **Derivatives Domain (`Logistic Regression`):** Binance Vision metrics such as funding rates, open interest, and long/short liquidations.
3. **Technicals Domain (`Logistic Regression`):** Yahoo Finance metrics such as RSI14, ATR14 percentage, volume ratio over 20 days, and closing spot price.  
> **Note:** Non-stationary price and on-chain features are transformed by fractional differentiation, where the optimal $d$ is evaluated separately for each feature, to create stationary ones while preserving memory. 


### Stacking Meta-Learner Architecture

* **Level-0 Domain Learners:** Fitting three base pipelines independently on their domain features to prevent cross-domain feature noise.
* **Out-Of-Fold (OOF) Probability Generation:** Training via time-series splitting and then generating out-of-fold probability vectors, where the vector components consist of the probability predictions from the base pipelines.
* **Level-1 Meta-Learner (`Logistic Regression`):** Using the base pipelines' probability predictions to optimize its own weight parameters and generating the final probability prediction. 

## 4. Execution & Risk Rules 

The raw probability predictions are translated into the portfolio allocation via the following mathematical rules.

1. **Probability Smoothing & Dynamic Sizing**

The meta-learner predictions are smoothed via an Exponential Moving Average to prevent signal noise:

$$\bar{P}_t = \text{EMA}(P(y=1)_t, \text{span}=2)$$

If $\bar{P}_t \ge $ \text{`Minimum signal probability floor`} ($0.55$ in the backtest), target allocation scales linearly up to $1.0$ (100% BTC), depending on the `Position scaling multiplier` ($2.50$ in the backtest):

$$\text{Target Allocation}_t = \min\left(1.0, \max\left(0.0, 2.50 \times (\bar{P}_t - 0.55)\right)\right)$$

2. **Overbought Momentum Gate (RSI Filter)**

The safeguard against entering during the blow-off tops. If $\text{RSI}_{14, t} > \text{`RSI cutoff`}$ ($75.0$ in the backtest), the target allocation is forced to 0.0% (All cash). 

3. **Dynamic ATR Trailing Stop**

A continuous trailing stop loss tracks peak close prices for active positions since entry: 

$\text{Stop Price}_t = \text{Peak Price}_t - (\text{`ATR Multiplier`} \times \text{ATR}_{14, t-1})$

If $\text{Close}_t \le \text{Stop Price}_t$, the position is completely liquidated to cash (`ATR Multiplier` = 1.0 in the backtest).

4. **Rebalance Threshold & Friction Drag**

* The execution at time $t$ uses the signals at time $t-1$ to prevent a look-ahead bias.
* If the position sizing changes by less than $4\%$ (min_rebalance_delta=0.04), there will be no rebalance in position to reduce unnecessary portfolio churn. 
* All trades in backtesting include 0.08% total friction (0.06% exchange taker fee + 0.02% estimated market impact) to simulate the real market fee and friction.

---

## 5. Setup & Installation

### Prerequisites

* Python 3.10+
* Virtual environment tool (`venv` or `conda`)

```bash
# Clone repository
git clone [https://github.com/your-username/btc_stacking_system.git](https://github.com/your-username/btc_stacking_system.git)
cd btc_stacking_system

# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

---

## 6. Usage & Pipeline Execution

The pipeline is managed via `main.py` CLI interface:

Run Complete End-to-End Pipeline

```bash
python main.py --all
```

Modular Pipeline Commands

```bash
# 1. Fetch multi-source raw market data
python main.py --download-data

# 2. Build fractional differentiation feature matrix
python main.py --features-building

# 3. Train Level-0 and Level-1 stacking ensemble
python main.py --train

# 4. Execute vectorized backtest engine
python main.py --backtest
```

Interactive Streamlit Dashboard
* The dashboard can be used to explore equity curves, parameter grid-search plateaus, and performance metrics:

```bash
streamlit run app.py
```

---

## 7. Output Artifacts

The pipeline execution gives out the following output artifacts:

* Holdout Predictions: `data/processed/holdout_predictions.csv` (The meta-learner prediction probabilities and actual outcomes).
* Trained Model Binary: `models/stacking_ensemble.pkl` (Saved model for back and forward tests).
* Equity Curve Plot: `data/processed/equity_curve.png` (Two-panel chart showing portfolio equity vs. benchmark and dynamic position allocation).

