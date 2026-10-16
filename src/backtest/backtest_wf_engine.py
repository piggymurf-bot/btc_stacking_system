import pandas as pd
import numpy as np

def run_backtest_walk_forward():
    # 1. Load your OOS predictions
    alpha_df = pd.read_csv("data/processed/alpha_wfo_out_of_sample_predictions.csv")
    volat_df = pd.read_csv("data/processed/volat_wfo_out_of_sample_predictions.csv")

    alpha_df["date"] = pd.to_datetime(alpha_df["date"])
    volat_df["date"] = pd.to_datetime(volat_df["date"])

    # Merge both engines on date
    merged_df = pd.merge(
        alpha_df[["date", "y_pred", "y_true"]], 
        volat_df[["date", "y_pred"]], 
        on="date", 
        suffixes=("_alpha", "_vol")
    ).sort_values("date").reset_index(drop=True)
   
    # Rename for clarity
    merged_df = merged_df.rename(columns={
        "y_pred_alpha": "alpha_pred",
        "y_pred_vol": "vol_pred",
        "y_true": "actual_return_14d"
    })

    # 2. Construct Position Sizing on the whole dataframe first
    merged_df["vol_pred"] = merged_df["vol_pred"].clip(lower=0.01)
    merged_df["raw_position"] = merged_df["alpha_pred"] / (merged_df["vol_pred"] ** 2)

    alpha_upper = merged_df["alpha_pred"].quantile(0.70)
    alpha_lower = merged_df["alpha_pred"].quantile(0.30)
    
    # Test stricter, asymmetric or symmetric high-conviction thresholds
    # alpha_upper = merged_df["alpha_pred"].quantile(0.85)  # Top 15% for longs
    # alpha_lower = merged_df["alpha_pred"].quantile(0.15)  # Bottom 15% for shorts
    
    # Test a balanced middle-ground threshold (Top/Bottom 20%)
    # alpha_upper = merged_df["alpha_pred"].quantile(0.80)  # Top 20% for longs
    # alpha_lower = merged_df["alpha_pred"].quantile(0.20)  # Bottom 20% for shorts
    
    merged_df["position"] = 0.0
    merged_df.loc[merged_df["alpha_pred"] > alpha_upper, "position"] = merged_df["raw_position"]
    merged_df.loc[merged_df["alpha_pred"] < alpha_lower, "position"] = merged_df["raw_position"]

    MAX_LEVERAGE = 1.0
    merged_df["position"] = merged_df["position"].clip(-MAX_LEVERAGE, MAX_LEVERAGE)

    # 3. ENFORCE NON-OVERLAPPING 14-DAY SAMPLING (The Fix)
    # Sample every 14 rows to ensure trades and outcomes are completely independent
    rebalance_df = merged_df.iloc[::14].copy().reset_index(drop=True)

    # Calculate turnover and transaction costs on the rebalance schedule
    rebalance_df["position_change"] = rebalance_df["position"].diff().abs().fillna(rebalance_df["position"].abs())
    TRANSACTION_COST = 0.001  # 10 bps per unit of position change
    rebalance_df["t_cost"] = rebalance_df["position_change"] * TRANSACTION_COST

    # Strategy PnL for each independent 14-day holding period
    rebalance_df["strategy_pnl"] = (rebalance_df["position"] * rebalance_df["actual_return_14d"]) - rebalance_df["t_cost"]
    rebalance_df["cumulative_pnl"] = rebalance_df["strategy_pnl"].cumsum()

    # 4. Compute True Independent Performance Metrics
    total_trades = (rebalance_df["position_change"] > 0).sum()
    win_rate = (rebalance_df["strategy_pnl"] > 0).mean() * 100
    mean_pnl = rebalance_df["strategy_pnl"].mean()
    std_pnl = rebalance_df["strategy_pnl"].std()
    
    # Correct Annualization for 14-day periods (there are 365 / 14 periods in a year)
    periods_per_year = 365 / 14
    sharpe_ratio = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(periods_per_year)

    # Maximum Drawdown calculation
    rolling_max = rebalance_df["cumulative_pnl"].cummax()
    drawdown = rebalance_df["cumulative_pnl"] - rolling_max
    max_drawdown = drawdown.min()

    print("\n========================================")
    print("📈 TRUE NON-OVERLAPPING BACKTEST PERFORMANCE")
    print(f"📊 Total Independent 14d Trades     : {total_trades}")
    print(f"🎯 Strategy Win Rate                : {win_rate:.2f}%")
    print(f"🚀 Annualized Sharpe Ratio          : {sharpe_ratio:.2f}")
    print(f"📉 Maximum Drawdown                 : {max_drawdown:.4f}")
    print("========================================")

    # Save backtest results
    output_backtest = "data/processed/portfolio_backtest_results_non_overlapping.csv"
    rebalance_df.to_csv(output_backtest, index=False)
    print(f"📁 Backtest log saved to: {output_backtest}")
    
    
    results = []
    # Test threshold quantiles from 0.50 (50/50 split) up to 0.85 (extreme tails)
    quantiles_to_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for q in quantiles_to_test:
        df_test = merged_df.copy()
    
        alpha_upper = df_test["alpha_pred"].quantile(q)
        alpha_lower = df_test["alpha_pred"].quantile(1 - q)
    
        df_test["position"] = 0.0
        df_test.loc[df_test["alpha_pred"] > alpha_upper, "position"] = df_test["raw_position"]
        df_test.loc[df_test["alpha_pred"] < alpha_lower, "position"] = df_test["raw_position"]
    
        MAX_LEVERAGE = 1.0
        df_test["position"] = df_test["position"].clip(-MAX_LEVERAGE, MAX_LEVERAGE)
    
        # 14-day non-overlapping sampling
        rebalance_df = df_test.iloc[::14].copy().reset_index(drop=True)
        rebalance_df["position_change"] = rebalance_df["position"].diff().abs().fillna(rebalance_df["position"].abs())
        rebalance_df["t_cost"] = rebalance_df["position_change"] * 0.001
    
        rebalance_df["strategy_pnl"] = (rebalance_df["position"] * rebalance_df["actual_return_14d"]) - rebalance_df["t_cost"]
        rebalance_df["cumulative_pnl"] = rebalance_df["strategy_pnl"].cumsum()
    
        total_trades = (rebalance_df["position_change"] > 0).sum()
        if total_trades == 0:
            continue
        
        win_rate = (rebalance_df["strategy_pnl"] > 0).mean() * 100
        mean_pnl = rebalance_df["strategy_pnl"].mean()
        std_pnl = rebalance_df["strategy_pnl"].std()
        
        periods_per_year = 365 / 14
        sharpe_ratio = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(periods_per_year)
    
        rolling_max = rebalance_df["cumulative_pnl"].cummax()
        max_drawdown = (rebalance_df["cumulative_pnl"] - rolling_max).min()
        
        results.append({
            "Quantile": f"{int(q*100)}% / {int((1-q)*100)}%",
            "Trades": total_trades,
            "Win Rate (%)": round(win_rate, 2),
            "Sharpe": round(sharpe_ratio, 2),
            "Max DD (%)": round(max_drawdown * 100, 2)
        })

    results_df = pd.DataFrame(results)
    print("\n=== CONVICTION THRESHOLD GRID SEARCH ===")
    print(results_df.to_string(index=False))
    
if __name__ == "__main__":
    run_backtest_walk_forward()