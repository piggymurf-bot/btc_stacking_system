import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest.engine import VectorizedBacktester


class SensitivityAnalyzer:

    def __init__(self, preds_path="data/processed/holdout_predictions.csv"):
        self.preds_path = preds_path
        if not os.path.exists(self.preds_path):
            raise FileNotFoundError(f"Predictions file not found at: {self.preds_path}")
        self.tester = VectorizedBacktester(predictions_path=self.preds_path)

    def run_2d_grid(
        self,
        param_x="atr_multiplier",
        x_range=np.linspace(0.5, 2.5, 9),
        param_y="max_position_scale",
        y_range=np.linspace(1.5, 5.0, 8),
        fixed_params=None,
        metric="Annualized Sharpe Ratio",
    ):
        """Sweeps any two dynamic parameters across a grid and computes performance metrics."""
        
        # Default fixed baseline values for parameters not being swept
        base_kwargs = {
            "min_probability": 0.53,
            "max_position_scale": 3.0,
            "rsi_max_filter": 70.0,
            "atr_multiplier": 1.0,
            "smooth_window": 2,
            "min_rebalance_delta": 0.04,
            "sizing_mode": "dynamic",
        }
        
        if fixed_params:
            base_kwargs.update(fixed_params)

        results = []

        print(f"🔍 Sweeping Grid: {param_x} ({len(x_range)}) × {param_y} ({len(y_range)})...")

        for val_y in y_range:
            for val_x in x_range:
                # Copy baseline settings and assign the current grid values
                run_kwargs = base_kwargs.copy()
                run_kwargs[param_x] = float(val_x)
                run_kwargs[param_y] = float(val_y)

                self.tester.run_backtest(**run_kwargs)
                metrics = self.tester.compute_performance_metrics()

                results.append({
                    param_y: round(float(val_y), 2),
                    param_x: round(float(val_x), 2),
                    metric: metrics.get(metric, 0.0),
                    "Max Drawdown (%)": metrics.get("Max Drawdown (%)", 0.0),
                    "Total Return (%)": metrics.get("Total Return Strategy (%)", 0.0),
                })

        df_results = pd.DataFrame(results)

        # Pivot table for Plotly / Seaborn heatmap rendering
        pivot_metric = df_results.pivot(
            index=param_y, columns=param_x, values=metric
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