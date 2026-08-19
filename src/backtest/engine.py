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
      trading_fee: float = 0.0006,  # 0.06% exchange taker fee
      slippage: float = 0.0002,  # 0.02% estimated market impact
  ):
    if not os.path.exists(predictions_path):
      raise FileNotFoundError(
          f"Predictions file not found at {predictions_path}. "
          "Please run --train first to generate holdout predictions."
      )
    self.df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    self.initial_capital = initial_capital
    self.total_cost = trading_fee + slippage
    self.results_df = None

  def _compute_indicators(
      self, df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14
  ) -> pd.DataFrame:
    """Calculates RSI and ATR indicators on close prices."""
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
      sizing_mode: str = "dynamic",  # Options: 'dynamic' or 'binary'
      min_probability: float = 0.55,  # Floor where position sizing starts (>0)
      max_position_scale: float = 2.5,  # Scale factor: min(1.0, scale * (P - min_prob))
      rsi_max_filter: float = 65.0,  # Zero out exposure when RSI > 65
      atr_multiplier: float = 0.8,  # Dynamic ATR Trailing Stop Multiplier
      smooth_window: int = 2,  # EMA smoothing to reduce daily noise
      min_rebalance_delta: float = 0.04,  # Ignore position changes < 4%
  ) -> pd.DataFrame:
    """Executes backtest with Dynamic Sizing, RSI Guard, and ATR Trailing Stops."""
    df = self.df.copy()
    df = self._compute_indicators(df)

    # 1. Smooth Model Probabilities
    if smooth_window > 1:
      df["prob_smooth"] = df["y_pred_prob"].ewm(span=smooth_window).mean()
    else:
      df["prob_smooth"] = df["y_pred_prob"]

    # 2. Compute Raw Sizing Targets
    if sizing_mode == "dynamic":
      raw_scale = max_position_scale * (df["prob_smooth"] - min_probability)
      target_sizes = np.clip(raw_scale, 0.0, 1.0)
    else:
      target_sizes = np.where(df["prob_smooth"] >= 0.60, 1.0, 0.0)
    
    ## Macro Regime Switch: Check if price is above 200-day trend
    #macro_bullish = df["close"] > df["close"].rolling(200).mean()

    ## Convex Position Allocation
    ## In Bull Macro: Instant 100% exposure if prob > 0.52
    ## In Bear Macro: Strict probability requirement & reduced max exposure
    #target_sizes = np.where(    
    #    macro_bullish,
    #    np.where(df["prob_smooth"] >= 0.52, 1.0, 0.0),  # Full upside tracking
    #    np.where(
    #        df["prob_smooth"] >= 0.60, 0.30, 0.0
    #       ),  # Defensive cash preservation
    #    )
    
    # Apply RSI Overbought Gate: Force position target to 0 when RSI > rsi_max_filter
    target_sizes = np.where(df["rsi_14"] > rsi_max_filter, 0.0, target_sizes)
    df["raw_target_size"] = target_sizes

    # 3. Iterative Simulation with Rebalancing & Trailing Stops
    n = len(df)
    executed_positions = np.zeros(n)
    strategy_net_returns = np.zeros(n)
    turnover_history = np.zeros(n)

    prices = df["close"].values
    atrs = df["atr_14"].values

    in_position = False
    active_size = 0.0
    highest_price = 0.0

    for i in range(1, n):
      prev_target = target_sizes[i - 1]  # Yesterday's signal (no look-ahead)
      curr_price = prices[i]
      prev_price = prices[i - 1]
      curr_atr = atrs[i - 1]  # Yesterday's ATR

      if in_position:
        highest_price = max(highest_price, curr_price)
        stop_price = highest_price - (atr_multiplier * curr_atr)

        # Event A: ATR Trailing Stop Triggered
        if curr_price <= stop_price:
          in_position = False
          # Realize return down to stop price minus exit friction
          exit_turnover = active_size
          raw_return = active_size * ((stop_price - prev_price) / prev_price)
          cost = exit_turnover * self.total_cost

          strategy_net_returns[i] = raw_return - cost
          turnover_history[i] = exit_turnover
          active_size = 0.0
          executed_positions[i] = 0.0

        # Event B: Model Signal / RSI Exit (Target dropped to 0)
        elif prev_target == 0.0:
          in_position = False
          exit_turnover = active_size
          raw_return = active_size * ((curr_price - prev_price) / prev_price)
          cost = exit_turnover * self.total_cost

          strategy_net_returns[i] = raw_return - cost
          turnover_history[i] = exit_turnover
          active_size = 0.0
          executed_positions[i] = 0.0

        # Event C: Rebalance Fractional Position Size
        else:
          desired_size = prev_target
          size_change = abs(desired_size - active_size)

          # Only pay turnover fee if change exceeds rebalance threshold
          if size_change >= min_rebalance_delta:
            turnover = size_change
            active_size = desired_size
          else:
            turnover = 0.0

          raw_return = active_size * ((curr_price - prev_price) / prev_price)
          cost = turnover * self.total_cost

          strategy_net_returns[i] = raw_return - cost
          turnover_history[i] = turnover
          executed_positions[i] = active_size

      else:
        # Event D: New Entry from Cash
        if prev_target > 0.0:
          in_position = True
          active_size = prev_target
          highest_price = curr_price

          entry_turnover = active_size
          raw_return = active_size * ((curr_price - prev_price) / prev_price)
          cost = entry_turnover * self.total_cost

          strategy_net_returns[i] = raw_return - cost
          turnover_history[i] = entry_turnover
          executed_positions[i] = active_size

    df["position"] = executed_positions
    df["trades"] = turnover_history
    df["strategy_net_return"] = strategy_net_returns
    df["asset_return"] = df["close"].pct_change().fillna(0.0)

    # 4. Compute Cumulative Equity Curves
    df["equity_benchmark"] = self.initial_capital * (
        1 + df["asset_return"]
    ).cumprod()
    df["equity_strategy"] = self.initial_capital * (
        1 + df["strategy_net_return"]
    ).cumprod()

    self.results_df = df
    return df

  def compute_performance_metrics(self) -> dict:
    """Computes risk-adjusted performance indicators."""
    if self.results_df is None:
      raise ValueError("Run backtest before computing metrics.")

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

    active_days = df[df["position"] > 0.0]
    win_rate = (
        (active_days["strategy_net_return"] > 0).mean()
        if len(active_days) > 0
        else 0.0
    )

    total_turnover = df["trades"].sum()

    return {
        "Total Return Strategy (%)": round(total_strat_return * 100, 2),
        "Total Return BTC Benchmark (%)": round(total_bench_return * 100, 2),
        "Annualized Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Total Portfolio Turnover (x)": round(total_turnover, 2),
    }

  def plot_equity_curve(
      self, save_path: str = "data/processed/equity_curve.png"
  ):
    """Saves strategy equity curve and position allocation plot."""
    if self.results_df is None:
      raise ValueError("Run backtest before plotting.")

    df = self.results_df
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Upper Plot: Equity Curves
    ax1.plot(
        df.index,
        df["equity_strategy"],
        label="Integrated Dynamic Sizing + ATR Stop",
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
        "Integrated Dynamic Sizing & Risk Controls vs. BTC Benchmark",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Lower Plot: Scaled Active Position Exposure
    ax2.fill_between(
        df.index,
        df["position"],
        step="pre",
        alpha=0.4,
        color="#1f77b4",
        label="Active Portfolio Allocation (0.0 to 1.0)",
    )
    ax2.set_ylabel("Position Size")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Equity curve plot saved to {save_path}")


def run_backtest_pipeline(long_threshold: float = 0.60):
  """Execution wrapper called by main.py."""
  print("=== Running Integrated Dynamic Sizing + ATR Stop Backtest Engine ===")
  tester = VectorizedBacktester()

  # Run Integrated Dynamic Strategy
  #tester.run_backtest(
  #    sizing_mode="dynamic",
  #    min_probability=0.55,  # Higher floor eliminates noise
  #    max_position_scale=2.5,
  #    rsi_max_filter=65.0,
  #    atr_multiplier=0.8,
  #)
  
  # Aggressive / Less Defensive Configuration
  tester.run_backtest(
      sizing_mode="dynamic",
      min_probability=0.51,  # Lower entry floor (was 0.55)
      max_position_scale=3.5,  # Faster scaling to 1.0 position size (was 2.5)
      rsi_max_filter=72.0,  # Allow holding during strong momentum (was 65.0)
      atr_multiplier=1.5,  # Wider stop to survive standard BTC volatility (was 0.8)
      smooth_window=2,  # Keep fast reactivity
      min_rebalance_delta=0.04,
  )
  
  metrics = tester.compute_performance_metrics()

  print("\n--- Strategy Performance Results ---")
  for key, value in metrics.items():
    print(f"{key:32s}: {value}")

  tester.plot_equity_curve()