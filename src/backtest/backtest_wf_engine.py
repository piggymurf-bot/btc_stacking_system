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
    
    
    # Filter to active trades (where position != 0)
    trades_df = rebalance_df[rebalance_df["position"] != 0].copy()
    print(f"Total active trades: {len(trades_df)}")
    print(trades_df[["date", "position", "alpha_pred", "actual_return_14d", "strategy_pnl"]].to_string())
    
    # results = []
    # # Test threshold quantiles from 0.50 (50/50 split) up to 0.85 (extreme tails)
    # quantiles_to_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    # for q in quantiles_to_test:
    #     df_test = merged_df.copy()
    
    #     alpha_upper = df_test["alpha_pred"].quantile(q)
    #     alpha_lower = df_test["alpha_pred"].quantile(1 - q)
    
    #     df_test["position"] = 0.0
    #     df_test.loc[df_test["alpha_pred"] > alpha_upper, "position"] = df_test["raw_position"]
    #     df_test.loc[df_test["alpha_pred"] < alpha_lower, "position"] = df_test["raw_position"]
    
    #     MAX_LEVERAGE = 1.0
    #     df_test["position"] = df_test["position"].clip(-MAX_LEVERAGE, MAX_LEVERAGE)
    
    #     # 14-day non-overlapping sampling
    #     rebalance_df = df_test.iloc[::14].copy().reset_index(drop=True)
    #     rebalance_df["position_change"] = rebalance_df["position"].diff().abs().fillna(rebalance_df["position"].abs())
    #     rebalance_df["t_cost"] = rebalance_df["position_change"] * 0.001
    
    #     rebalance_df["strategy_pnl"] = (rebalance_df["position"] * rebalance_df["actual_return_14d"]) - rebalance_df["t_cost"]
    #     rebalance_df["cumulative_pnl"] = rebalance_df["strategy_pnl"].cumsum()
    
    #     total_trades = (rebalance_df["position_change"] > 0).sum()
    #     if total_trades == 0:
    #         continue
        
    #     win_rate = (rebalance_df["strategy_pnl"] > 0).mean() * 100
    #     mean_pnl = rebalance_df["strategy_pnl"].mean()
    #     std_pnl = rebalance_df["strategy_pnl"].std()
        
    #     periods_per_year = 365 / 14
    #     sharpe_ratio = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(periods_per_year)
    
    #     rolling_max = rebalance_df["cumulative_pnl"].cummax()
    #     max_drawdown = (rebalance_df["cumulative_pnl"] - rolling_max).min()
        
    #     results.append({
    #         "Quantile": f"{int(q*100)}% / {int((1-q)*100)}%",
    #         "Trades": total_trades,
    #         "Win Rate (%)": round(win_rate, 2),
    #         "Sharpe": round(sharpe_ratio, 2),
    #         "Max DD (%)": round(max_drawdown * 100, 2)
    #     })

    # results_df = pd.DataFrame(results)
    # print("\n=== CONVICTION THRESHOLD GRID SEARCH ===")
    # print(results_df.to_string(index=False))

def run_backtest_with_sltp():
    # Let's run a slippage stress test on our robust production candidates:
    # Candidate A: 80%/19% quantile, 8% SL, 10% TP
    # Candidate B: 75%/25% quantile, 8% SL, 10% TP
    # We will test normal cost (10 bps = 0.001) vs stressed cost (20 bps = 0.002 and 30 bps = 0.003)

    master_df = pd.read_csv("data/processed/master_feature_addlhar_alpha.csv")
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df = master_df.sort_values("date").reset_index(drop=True)

    alpha_df = pd.read_csv("data/processed/alpha_wfo_out_of_sample_predictions.csv")
    volat_df = pd.read_csv("data/processed/volat_wfo_out_of_sample_predictions.csv")

    alpha_df["date"] = pd.to_datetime(alpha_df["date"])
    volat_df["date"] = pd.to_datetime(volat_df["date"])

    merged_df = pd.merge(
        alpha_df[["date", "y_pred", "y_true"]], 
        volat_df[["date", "y_pred"]], 
        on="date", 
        suffixes=("_alpha", "_vol")
    ).sort_values("date").reset_index(drop=True)
   
    merged_df = merged_df.rename(columns={
        "y_pred_alpha": "alpha_pred",
        "y_pred_vol": "vol_pred",
        "y_true": "actual_return_14d"
    })

    merged_df = pd.merge(merged_df, master_df[["date", "open", "high", "low", "close"]], on="date", how="inner")
    merged_df["vol_pred"] = merged_df["vol_pred"].clip(lower=0.01)

    rebalance_indices = list(range(0, len(merged_df), 14))

    def evaluate_candidate(q, sl, tp, t_cost_rate):
        history_alphas = []
        trade_returns = []
        
        for i, idx in enumerate(rebalance_indices):
            row = merged_df.iloc[idx]
            current_alpha = row["alpha_pred"]
            history_alphas.append(current_alpha)
        
            if i < 5:
                continue
            
            past_alphas = pd.Series(history_alphas[:-1])
            alpha_upper = past_alphas.quantile(q)
            alpha_lower = past_alphas.quantile(1 - q)
        
            raw_pos = current_alpha / (row["vol_pred"] ** 2)
            pos = 0.0
            if current_alpha > alpha_upper:
                pos = raw_pos
            elif current_alpha < alpha_lower:
                pos = raw_pos
            pos = np.clip(pos, -1.0, 1.0)
        
            if pos == 0:
                continue
            
            entry_price = row["close"]
            end_idx = min(idx + 14, len(merged_df) - 1)
            holding_period = merged_df.iloc[idx + 1 : end_idx + 1]
        
            exit_price = holding_period.iloc[-1]["close"]
        
            for _, day_row in holding_period.iterrows():
                if pos > 0:
                    if day_row["low"] <= entry_price * (1 - sl):
                        exit_price = entry_price * (1 - sl)
                        break
                    elif day_row["high"] >= entry_price * (1 + tp):
                        exit_price = entry_price * (1 + tp)
                        break
                else:
                    if day_row["high"] >= entry_price * (1 + sl):
                        exit_price = entry_price * (1 + sl)
                        break
                    elif day_row["low"] <= entry_price * (1 - tp):
                        exit_price = entry_price * (1 - tp)
                        break
                    
            if pos > 0:
                trade_return = (exit_price - entry_price) / entry_price
            else:
                trade_return = (entry_price - exit_price) / entry_price
            
            t_cost = abs(pos) * t_cost_rate
            strategy_pnl = (pos * trade_return) - t_cost
            trade_returns.append(strategy_pnl)
        
        trades_arr = np.array(trade_returns)
        cum_pnl = trades_arr.cumsum()
        win_rate = (trades_arr > 0).mean() * 100
        mean_pnl = trades_arr.mean()
        std_pnl = trades_arr.std()
        sharpe = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(365 / 14)
    
        rolling_max = np.maximum.accumulate(cum_pnl)
        max_dd = (cum_pnl - rolling_max).min()
    
        return len(trades_arr), win_rate, sharpe, max_dd * 100

    stress_results = []
    candidates = [
        ("80%/19%", 0.80, 0.08, 0.10),
        ("75%/25%", 0.75, 0.08, 0.10)
    ]

    for name, q, sl, tp in candidates:
        for cost_label, cost_val in [("10 bps (Normal)", 0.001), ("20 bps (Stressed)", 0.002), ("30 bps (Severe)", 0.003)]:
            trades, wr, sharpe, mdd = evaluate_candidate(q, sl, tp, cost_val)
            stress_results.append({
                "Candidate": name,
                "Slippage/Cost": cost_label,
                "Trades": trades,
                "Win Rate (%)": round(wr, 2),
                "Sharpe": round(sharpe, 2),
                "Max DD (%)": round(mdd, 2)
            })

    stress_df = pd.DataFrame(stress_results)
    print(stress_df.to_string(index=False))

def run_backtest_with_rolling_quantiles():
    # 1. Load data
    master_df = pd.read_csv("data/processed/master_feature_addlhar_alpha.csv")
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df = master_df.sort_values("date").reset_index(drop=True)

    alpha_df = pd.read_csv("data/processed/alpha_wfo_out_of_sample_predictions.csv")
    volat_df = pd.read_csv("data/processed/volat_wfo_out_of_sample_predictions.csv")

    alpha_df["date"] = pd.to_datetime(alpha_df["date"])
    volat_df["date"] = pd.to_datetime(volat_df["date"])

    merged_df = pd.merge(
        alpha_df[["date", "y_pred", "y_true"]], 
        volat_df[["date", "y_pred"]], 
        on="date", 
        suffixes=("_alpha", "_vol")
    ).sort_values("date").reset_index(drop=True)
   
    merged_df = merged_df.rename(columns={
        "y_pred_alpha": "alpha_pred",
        "y_pred_vol": "vol_pred",
        "y_true": "actual_return_14d"
    })

    merged_df = pd.merge(merged_df, master_df[["date", "open", "high", "low", "close"]], on="date", how="inner")
    merged_df["vol_pred"] = merged_df["vol_pred"].clip(lower=0.01)

    rebalance_indices = range(0, len(merged_df), 14)
    trade_results = []
    
    STOP_LOSS_PCT = 0.08    
    TAKE_PROFIT_PCT = 0.10  
    TRANSACTION_COST = 0.001

    # We need a warm-up window of past predictions to establish initial quantiles (e.g., first 5 rebalance periods)
    history_alphas = []

    for i, idx in enumerate(rebalance_indices):
        row = merged_df.iloc[idx]
        current_alpha = row["alpha_pred"]
        
        # Add current alpha to history for future steps
        history_alphas.append(current_alpha)
        
        # Warm-up phase: need at least 5 historical points to calculate quantiles
        if i < 5:
            continue
            
        # Compute thresholds strictly from PAST alpha predictions (Expanding Window)
        past_alphas = pd.Series(history_alphas[:-1]) # Exclude current to avoid lookahead
        alpha_upper = past_alphas.quantile(0.70)
        alpha_lower = past_alphas.quantile(0.30)
        
        # Calculate raw position
        raw_pos = current_alpha / (row["vol_pred"] ** 2)
        pos = 0.0
        if current_alpha > alpha_upper:
            pos = raw_pos
        elif current_alpha < alpha_lower:
            pos = raw_pos
            
        pos = np.clip(pos, -1.0, 1.0)
        if pos == 0:
            continue
            
        entry_date = row["date"]
        entry_price = row["close"]
        
        end_idx = min(idx + 14, len(merged_df) - 1)
        holding_period = merged_df.iloc[idx + 1 : end_idx + 1]
        
        exit_price = holding_period.iloc[-1]["close"]
        exit_reason = "time_expiration"
        
        for _, day_row in holding_period.iterrows():
            if pos > 0:
                if day_row["low"] <= entry_price * (1 - STOP_LOSS_PCT):
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                    exit_reason = "stop_loss"
                    break
                elif day_row["high"] >= entry_price * (1 + TAKE_PROFIT_PCT):
                    exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    exit_reason = "take_profit"
                    break
            else:
                if day_row["high"] >= entry_price * (1 + STOP_LOSS_PCT):
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
                    exit_reason = "stop_loss"
                    break
                elif day_row["low"] <= entry_price * (1 - TAKE_PROFIT_PCT):
                    exit_price = entry_price * (1 - TAKE_PROFIT_PCT)
                    exit_reason = "take_profit"
                    break
                
        if pos > 0:
            trade_return = (exit_price - entry_price) / entry_price
        else:
            trade_return = (entry_price - exit_price) / entry_price
            
        t_cost = abs(pos) * TRANSACTION_COST
        strategy_pnl = (pos * trade_return) - t_cost
        
        trade_results.append({
            "entry_date": entry_date,
            "position": pos,
            "exit_reason": exit_reason,
            "trade_return": trade_return,
            "strategy_pnl": strategy_pnl
        })

    trades_df = pd.DataFrame(trade_results)
    trades_df["cumulative_pnl"] = trades_df["strategy_pnl"].cumsum()
    
    win_rate = (trades_df["strategy_pnl"] > 0).mean() * 100
    mean_pnl = trades_df["strategy_pnl"].mean()
    std_pnl = trades_df["strategy_pnl"].std()
    sharpe = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(365 / 14)
    
    rolling_max = trades_df["cumulative_pnl"].cummax()
    max_dd = (trades_df["cumulative_pnl"] - rolling_max).min()

    print("\n========================================")
    print("🔒 STRICT OOS ROLLING QUANTILE PERFORMANCE")
    print(f"📊 Total Trades                     : {len(trades_df)}")
    print(f"🎯 Strategy Win Rate                : {win_rate:.2f}%")
    print(f"🚀 Annualized Sharpe Ratio          : {sharpe:.2f}")
    print(f"📉 Maximum Drawdown                 : {max_dd:.4f}")
    print("========================================")
    

def run_rolling_grid_search():
    # 1. Load data
    master_df = pd.read_csv("data/processed/master_feature_addlhar_alpha.csv")
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df = master_df.sort_values("date").reset_index(drop=True)

    alpha_df = pd.read_csv("data/processed/alpha_wfo_out_of_sample_predictions.csv")
    volat_df = pd.read_csv("data/processed/volat_wfo_out_of_sample_predictions.csv")

    alpha_df["date"] = pd.to_datetime(alpha_df["date"])
    volat_df["date"] = pd.to_datetime(volat_df["date"])

    merged_df = pd.merge(
        alpha_df[["date", "y_pred", "y_true"]], 
        volat_df[["date", "y_pred"]], 
        on="date", 
        suffixes=("_alpha", "_vol")
    ).sort_values("date").reset_index(drop=True)
   
    merged_df = merged_df.rename(columns={
        "y_pred_alpha": "alpha_pred",
        "y_pred_vol": "vol_pred",
        "y_true": "actual_return_14d"
    })

    merged_df = pd.merge(merged_df, master_df[["date", "open", "high", "low", "close"]], on="date", how="inner")
    merged_df["vol_pred"] = merged_df["vol_pred"].clip(lower=0.01)

    rebalance_indices = list(range(0, len(merged_df), 14))
    TRANSACTION_COST = 0.001

    # Define Grid Parameters
    quantile_grids = [0.60, 0.65, 0.70, 0.75, 0.80]
    sl_grid = [0.05, 0.08, 0.10]
    tp_grid = [0.08, 0.10, 0.15]

    grid_results = []

    for q in quantile_grids:
        # Step A: Generate strict rolling signals for this quantile configuration
        history_alphas = []
        q_signals = {}
        
        for i, idx in enumerate(rebalance_indices):
            row = merged_df.iloc[idx]
            current_alpha = row["alpha_pred"]
            history_alphas.append(current_alpha)
            
            if i < 5:  # Warm-up window
                continue
                
            past_alphas = pd.Series(history_alphas[:-1])  # Strict OOS expanding window
            alpha_upper = past_alphas.quantile(q)
            alpha_lower = past_alphas.quantile(1 - q)
            
            raw_pos = current_alpha / (row["vol_pred"] ** 2)
            pos = 0.0
            if current_alpha > alpha_upper:
                pos = raw_pos
            elif current_alpha < alpha_lower:
                pos = raw_pos
            pos = np.clip(pos, -1.0, 1.0)
            
            q_signals[idx] = {
                "pos": pos,
                "entry_price": row["close"]
            }

        # Step B: Test SL and TP combinations on these rolling signals
        for sl in sl_grid:
            for tp in tp_grid:
                trade_returns = []
                
                for idx, sig in q_signals.items():
                    pos = sig["pos"]
                    if pos == 0:
                        continue
                    entry_price = sig["entry_price"]
                    
                    end_idx = min(idx + 14, len(merged_df) - 1)
                    holding_period = merged_df.iloc[idx + 1 : end_idx + 1]
                    
                    exit_price = holding_period.iloc[-1]["close"]
                    
                    for _, day_row in holding_period.iterrows():
                        if pos > 0:
                            if day_row["low"] <= entry_price * (1 - sl):
                                exit_price = entry_price * (1 - sl)
                                break
                            elif day_row["high"] >= entry_price * (1 + tp):
                                exit_price = entry_price * (1 + tp)
                                break
                        else:
                            if day_row["high"] >= entry_price * (1 + sl):
                                exit_price = entry_price * (1 + sl)
                                break
                            elif day_row["low"] <= entry_price * (1 - tp):
                                exit_price = entry_price * (1 - tp)
                                break
                                
                    if pos > 0:
                        trade_return = (exit_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - exit_price) / entry_price
                        
                    t_cost = abs(pos) * TRANSACTION_COST
                    strategy_pnl = (pos * trade_return) - t_cost
                    trade_returns.append(strategy_pnl)
                
                if len(trade_returns) == 0:
                    continue
                
                trades_arr = np.array(trade_returns)
                cum_pnl = trades_arr.cumsum()
                win_rate = (trades_arr > 0).mean() * 100
                mean_pnl = trades_arr.mean()
                std_pnl = trades_arr.std()
                sharpe = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(365 / 14)
                
                rolling_max = np.maximum.accumulate(cum_pnl)
                max_dd = (cum_pnl - rolling_max).min()
                
                grid_results.append({
                    "Quantile": f"{int(q*100)}%/{int((1-q)*100)}%",
                    "SL": f"{int(sl*100)}%",
                    "TP": f"{int(tp*100)}%",
                    "Trades": len(trades_arr),
                    "Win Rate (%)": round(win_rate, 2),
                    "Sharpe": round(sharpe, 2),
                    "Max DD (%)": round(max_dd * 100, 2)
                })

    res_df = pd.DataFrame(grid_results)
    print("\n=== STRICT OOS ROLLING GRID SEARCH RESULTS ===")
    print(res_df.sort_values(by="Sharpe", ascending=False).to_string(index=False))
    
    
if __name__ == "__main__":
    #run_backtest_walk_forward()
    run_backtest_with_sltp()
    #run_backtest_with_rolling_quantiles()
    #run_rolling_grid_search()
 
    