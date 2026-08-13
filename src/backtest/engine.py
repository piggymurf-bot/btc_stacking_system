# src/backtest/engine.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class VectorizedBacktester:

  def __init__(
      self,
      predictions_path: str = "data/processed/holdout_predictions.csv",
      initial_capital: float = 10000.0,
      trading_fee: float = 0.0006,  # 0.06% exchange fee
  ):
    if not os.path.exists(predictions_path):
      raise FileNotFoundError(
          f"Predictions file not found at {predictions_path}."
      )
    self.df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    self.initial_capital = initial_capital
    self.fee = trading_fee
    self.results_df = None

  def _compute_indicators(
      self, df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14
  ) -> pd.DataFrame:
    """Calculates RSI and ATR indicators on holdout close prices."""
    # 1. RSI (14-period)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # 2. True Range & ATR (14-period)
    prev_close = df["close"].shift(1)
    if "high" in df.columns and "low" in df.columns:
      tr1 = df["high"] - df["low"]
      tr2 = (df["high"] - prev_close).abs()
      tr3 = (df["low"] - prev_close).abs()
      df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
      df["true_range"] = (df["close"] - prev_close).abs()

    df["atr_14"] = (
        df["true_range"].rolling(window=atr_period, min_periods=1).mean()
    )
    return df

  def run_backtest(
      self,
      long_threshold: float = 0.60,
      rsi_max_filter: float = 65.0,  # Filter out buys when RSI > 65
      atr_multiplier: float = 0.8,  # Dynamic ATR Trailing Stop Multiplier
      use_atr_stop: bool = True,
  ) -> pd.DataFrame:
    """Executes backtest with Combined Entry Signals & ATR Trailing Stop-Loss."""
    df = self.df.copy()
    df = self._compute_indicators(df)

    # 1. Combined Entry Signal (ML Probability + RSI Filter)
    df["entry_signal"] = np.where(
        (df["y_pred_prob"] >= long_threshold)
        & (df["rsi_14"] <= rsi_max_filter),
        1,
        0,
    )

    n = len(df)
    positions = np.zeros(n)
    strategy_returns = np.zeros(n)

    in_position = False
    highest_price = 0.0

    prices = df["close"].values
    atrs = df["atr_14"].values
    signals = df["entry_signal"].values

    # 2. Iterative Day-by-Day Simulation Loop
    for i in range(1, n):
      prev_signal = signals[i - 1]  # Yesterday's signal (no look-ahead)
      curr_price = prices[i]
      prev_price = prices[i - 1]
      curr_atr = atrs[i - 1]  # Yesterday's ATR

      if in_position:
        highest_price = max(highest_price, curr_price)

        # Dynamic Stop Price
        if use_atr_stop:
          stop_price = highest_price - (atr_multiplier * curr_atr)
        else:
          stop_price = highest_price * (1 - 0.025)

        # Condition A: Trailing Stop Hit
        if curr_price <= stop_price:
          in_position = False
          positions[i] = 0
          day_return = ((stop_price - prev_price) / prev_price) - self.fee
          strategy_returns[i] = day_return

        # Condition B: ML Model Signal Turned Off
        elif prev_signal == 0:
          in_position = False
          positions[i] = 0
          day_return = ((curr_price - prev_price) / prev_price) - self.fee
          strategy_returns[i] = day_return

        # Condition C: Hold Position
        else:
          positions[i] = 1
          strategy_returns[i] = (curr_price - prev_price) / prev_price

      else:
        # Condition D: Enter Position
        if prev_signal == 1:
          in_position = True
          highest_price = curr_price
          positions[i] = 1
          strategy_returns[i] = (
              (curr_price - prev_price) / prev_price
          ) - self.fee

    df["position"] = positions
    df["strategy_net_return"] = strategy_returns
    df["asset_return"] = df["close"].pct_change().fillna(0)

    # 3. Equity Curves
    df["equity_benchmark"] = self.initial_capital * (
        1 + df["asset_return"]
    ).cumprod()
    df["equity_strategy"] = self.initial_capital * (
        1 + df["strategy_net_return"]
    ).cumprod()

    self.results_df = df
    return df

  def run_atr_sensitivity_scan(
      self, multipliers=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
  ):
    """Runs grid search across ATR multipliers to locate optimal trailing stop distance."""
    print("=== ATR MULTIPLIER SENSITIVITY ANALYSIS ===")
    results = []
    for mult in multipliers:
      self.run_backtest(atr_multiplier=mult, use_atr_stop=True)
      metrics = self.compute_performance_metrics()
      results.append({
          "ATR Multiplier": f"{mult}x",
          "Total Return (%)": metrics["Total Return Strategy (%)"],
          "Sharpe Ratio": metrics["Annualized Sharpe Ratio"],
          "Max Drawdown (%)": metrics["Max Drawdown (%)"],
          "Trades": metrics["Total Trades Executed"],
      })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df

  def compute_performance_metrics(self) -> dict:
    """Computes strategy risk-adjusted performance indicators."""
    df = self.results_df
    net_ret = df["strategy_net_return"]
    std_ret = net_ret.std()

    sharpe = np.sqrt(365) * (net_ret.mean() / std_ret) if std_ret != 0 else 0.0

    cum_eq = df["equity_strategy"]
    peak = cum_eq.cummax()
    drawdown = (cum_eq - peak) / peak
    max_drawdown = drawdown.min()

    total_strat_return = (cum_eq.iloc[-1] / self.initial_capital) - 1.0
    total_bench_return = (
        df["equity_benchmark"].iloc[-1] / self.initial_capital
    ) - 1.0

    trades = pd.Series(df["position"]).diff().abs().sum() / 2

    return {
        "Total Return Strategy (%)": round(total_strat_return * 100, 2),
        "Total Return BTC Benchmark (%)": round(total_bench_return * 100, 2),
        "Annualized Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Total Trades Executed": int(trades),
    }

  def plot_equity_curve(
      self, save_path: str = "data/processed/equity_curve.png"
  ):
    """Plots strategy equity curve vs BTC benchmark and active long position state."""
    if self.results_df is None:
      raise ValueError(
          "Backtest has not been executed yet. Call run_backtest() first."
      )

    df = self.results_df

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # 1. Equity Curves
    ax1.plot(
        df.index,
        df["equity_strategy"],
        label="Stacking Strategy (RSI + ATR Stop)",
        linewidth=2,
        color="#1f77b4",
    )
    ax1.plot(
        df.index,
        df["equity_benchmark"],
        label="BTC Benchmark",
        linestyle="--",
        color="#7f7f7f",
        alpha=0.8,
    )
    ax1.set_title(
        "Integrated RSI + ATR Trailing Stop Strategy vs. BTC Benchmark",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Active Position State
    ax2.fill_between(
        df.index,
        df["position"],
        step="pre",
        alpha=0.4,
        color="#1f77b4",
        label="Active Long Position (1 = Held, 0 = Cash)",
    )
    ax2.set_ylabel("Position State")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Equity curve plot saved to {save_path}")


def run_backtest_pipeline(long_threshold: float = 0.60):
  """Execution wrapper called by main.py CLI."""
  print("=== Running Integrated RSI + ATR Trailing Stop Backtest ===")
  tester = VectorizedBacktester()

  # Execute optimal production config: 0.8x ATR + RSI <= 65 Filter
  tester.run_backtest(
      long_threshold=long_threshold,
      rsi_max_filter=65.0,
      atr_multiplier=0.8,
      use_atr_stop=True,
  )

  metrics = tester.compute_performance_metrics()

  print("\n--- Final Backtest Performance Results (0.8x ATR) ---")
  for key, value in metrics.items():
    print(f"{key:32s}: {value}")

  tester.plot_equity_curve()