import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest.engine import VectorizedBacktester


class SensitivityAnalyzer:

  def __init__(
      self, preds_path="data/processed/holdout_predictions.csv"
  ):
    self.preds_path = preds_path
    if not os.path.exists(self.preds_path):
      raise FileNotFoundError(f"Predictions file not found at: {self.preds_path}")
    self.tester = VectorizedBacktester(predictions_path=self.preds_path)

  def run_2d_grid(
      self,
      scale_range=np.linspace(1.5, 5.0, 8),
      atr_range=np.linspace(0.5, 2.5, 9),
      fixed_min_prob=0.53,
      fixed_rsi_cap=70.0,
      metric="Annualized Sharpe Ratio",
  ):
    """Sweeps two key parameters across a grid and computes performance metrics."""
    results = []

    print(
        f"🔍 Sweeping Grid: {len(scale_range)} Scale levels × {len(atr_range)} ATR levels..."
    )

    for scale in scale_range:
      for atr in atr_range:
        self.tester.run_backtest(
            sizing_mode="dynamic",
            min_probability=fixed_min_prob,
            max_position_scale=float(scale),
            rsi_max_filter=fixed_rsi_cap,
            atr_multiplier=float(atr),
            smooth_window=2,
            min_rebalance_delta=0.04,
        )
        metrics = self.tester.compute_performance_metrics()
        results.append({
            "Position Scale": round(float(scale), 2),
            "ATR Multiplier": round(float(atr), 2),
            metric: metrics.get(metric, 0.0),
            "Max Drawdown (%)": metrics.get("Max Drawdown (%)", 0.0),
            "Total Return (%)": metrics.get("Total Return Strategy (%)", 0.0),
        })

    df_results = pd.DataFrame(results)

    # Pivot table for the selected metric
    pivot_metric = df_results.pivot(
        index="Position Scale", columns="ATR Multiplier", values=metric
    )

    return df_results, pivot_metric

  def plot_heatmap(
      self,
      pivot_table,
      metric_name="Annualized Sharpe Ratio",
      save_path=None,
  ):
    """Generates and displays a heatmap of the parameter grid."""
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": metric_name},
        linewidths=0.5,
    )

    plt.title(
        f"Parameter Stability Plateau: {metric_name}",
        fontsize=14,
        pad=15,
        weight="bold",
    )
    plt.xlabel("ATR Trailing Stop Multiplier", fontsize=11, labelpad=10)
    plt.ylabel("Position Scaling Multiplier", fontsize=11, labelpad=10)
    plt.gca().invert_yaxis()  # Standard axis orientation

    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300)
      print(f"📊 Heatmap saved to {save_path}")

    plt.show()


# -----------------------------------------------------------------------------
# STANDALONE EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
  analyzer = SensitivityAnalyzer()

  # Define ranges to test
  scales = np.linspace(1.5, 5.0, 8)  # 1.5 to 5.0
  atrs = np.linspace(0.5, 2.5, 9)  # 0.5 to 2.5

  # Run grid search for Sharpe Ratio
  df_raw, pivot_sharpe = analyzer.run_2d_grid(
      scale_range=scales, atr_range=atrs, metric="Annualized Sharpe Ratio"
  )

  # Display plot
  analyzer.plot_heatmap(pivot_sharpe, metric_name="Annualized Sharpe Ratio")