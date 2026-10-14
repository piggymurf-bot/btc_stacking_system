import os
import numpy as np
import pandas as pd

def compute_indicators(df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    prev_close = df["close"].shift(1)
    if "high" in df.columns and "low" in df.columns:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        df["true_range"] = (df["close"] - prev_close).abs()

    df["atr_14"] = df["true_range"].rolling(window=atr_period, min_periods=1).mean()
    return df

def run_backtest_pipeline(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    output_path: str = "data/processed/final_rigorous_hybrid_backtest.csv",
    #alpha_oof_path: str = "data/processed/oof_alpha.csv",
    #vol_oof_path: str = "data/processed/oof_volat.csv",
    alpha_oof_path: str = "data/processed/holdout_fold5_predictions_alpha.csv",
    vol_oof_path: str = "data/processed/holdout_fold5_predictions_volat.csv",
    alpha_target: str = "target_fwd_sharpe_7d",
    vol_target: str = "target_rv",
    rsi_max_filter: float = 75.0,
    atr_multiplier: float = 1.0,
    min_rebalance_delta: float = 0.05,  # Prevents excessive micro-turnover
    transaction_cost_bps: float = 10.0
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}.")

    print(f"📥 Loading master dataset from: {data_path}")
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError("Dataset must contain a 'close' column.")

    required_cols = [alpha_target, vol_target, "date", "close", "high", "low"] 
    model_df = df.dropna(subset=required_cols).reset_index(drop=True)

    oof_alpha = pd.read_csv(alpha_oof_path)
    oof_vol = pd.read_csv(vol_oof_path)

    model_df["oof_alpha_pred"] = oof_alpha["oof_alpha_preds"]
    model_df["oof_vol_pred"] = oof_vol["oof_volat_preds"]
    
    # Drop rows where OOF predictions haven't started yet
    model_df = model_df[(model_df["oof_alpha_pred"] != 0.0) & (model_df["oof_vol_pred"] != 0.0)].copy()
    
    backtest_df = model_df.dropna(subset=["oof_alpha_pred", "oof_vol_pred"]).copy()
    backtest_df = compute_indicators(backtest_df)

    print("\n⚖️ Applying Rigorous Non-Lookahead Execution Loop & Adaptive Tiers...")
    
    backtest_df["safe_vol"] = backtest_df["oof_vol_pred"].clip(lower=0.005)
    
    # --- Adaptive Conviction Thresholds (Tiered Sizing Rule without Look-Ahead) ---
    alpha_pred = backtest_df["oof_alpha_pred"]
    
    # Expanding window percentiles shifted by 1 day to prevent look-ahead bias
    p80 = alpha_pred.expanding(min_periods=30).quantile(0.80).shift(1)
    p30 = alpha_pred.expanding(min_periods=30).quantile(0.30).shift(1)

    raw_weights = np.zeros(len(backtest_df))

    # Tier 3: Bottom 30% Confidence (< 30th percentile) -> 0.0 exposure (Cash)
    # Tier 2: Middle 50% Confidence (30th to 80th percentile) -> 0.3 exposure
    mask_tier2 = (alpha_pred >= p30) & (alpha_pred < p80)
    raw_weights[mask_tier2] = 0.3

    # Tier 1: Top 20% Confidence (>= 80th percentile) -> 1.0 exposure
    mask_tier1 = alpha_pred >= p80
    raw_weights[mask_tier1] = 1.0

    # Risk Filters: Volatility threshold & RSI overbought gate
    backtest_df["rolling_vol_90"] = backtest_df["safe_vol"].expanding(min_periods=30).quantile(0.90)
    vol_threshold = backtest_df["rolling_vol_90"].shift(1)

    raw_weights[backtest_df["safe_vol"] > vol_threshold] = 0.0
    raw_weights[backtest_df["rsi_14"] > rsi_max_filter] = 0.0
    backtest_df["raw_target_size"] = raw_weights

    # Stateful Execution Simulation Loop (Strictly No Look-Ahead)
    n = len(backtest_df)
    executed_positions = np.zeros(n)
    strategy_net_returns = np.zeros(n)
    turnover_history = np.zeros(n)

    prices = backtest_df["close"].values
    atrs = backtest_df["atr_14"].values
    targets = backtest_df["raw_target_size"].values
    cost_rate = transaction_cost_bps / 10000.0

    in_position = False
    active_size = 0.0
    highest_price = 0.0

    for i in range(1, n):
        prev_target = targets[i - 1]  # Yesterday's tiered signal (no look-ahead)
        curr_price = prices[i]
        prev_price = prices[i - 1]
        curr_atr = atrs[i - 1]

        if in_position:
            stop_price = highest_price - (atr_multiplier * curr_atr)

            if curr_price <= stop_price:
                in_position = False
                exit_turnover = active_size
                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = exit_turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = exit_turnover
                active_size = 0.0
                executed_positions[i] = 0.0

            elif prev_target == 0.0:
                in_position = False
                exit_turnover = active_size
                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = exit_turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = exit_turnover
                active_size = 0.0
                executed_positions[i] = 0.0

            else:
                desired_size = prev_target
                size_change = abs(desired_size - active_size)

                if size_change >= min_rebalance_delta:
                    turnover = size_change
                    active_size = desired_size
                else:
                    turnover = 0.0

                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = turnover
                executed_positions[i] = active_size

                highest_price = max(highest_price, curr_price)

        else:
            if prev_target > 0.0:
                in_position = True
                active_size = prev_target
                highest_price = curr_price

                entry_turnover = active_size
                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = entry_turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = entry_turnover
                executed_positions[i] = active_size

    backtest_df["position_weight"] = executed_positions
    backtest_df["turnover"] = turnover_history
    backtest_df["net_strategy_return"] = strategy_net_returns
    backtest_df["benchmark_return"] = backtest_df["close"].pct_change().fillna(0.0)

    # --- Buy & Hold Benchmark Calculation ---
    backtest_df["benchmark_wealth"] = (1 + backtest_df["benchmark_return"]).cumprod()
    bench_final_wealth = backtest_df["benchmark_wealth"].iloc[-1]
    bench_cum_return = (bench_final_wealth - 1) * 100
    
    bench_rets = backtest_df["benchmark_return"].dropna()
    bench_ann_return = bench_rets.mean() * 365
    bench_ann_vol = bench_rets.std() * np.sqrt(365)
    bench_sharpe = bench_ann_return / bench_ann_vol if bench_ann_vol > 0 else 0.0
    
    bench_running_max = backtest_df["benchmark_wealth"].cummax()
    bench_max_dd = ((backtest_df["benchmark_wealth"] - bench_running_max) / bench_running_max).min()

    # Compute Final Performance Metrics
    clean_rets = backtest_df["net_strategy_return"].dropna()
    wealth = (1 + clean_rets).cumprod()
    running_max = wealth.cummax()
    max_dd = ((wealth - running_max) / running_max).min()
    ann_return = clean_rets.mean() * 365
    ann_vol = clean_rets.std() * np.sqrt(365)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    print("\n" + "="*60)
    print("🚀 ADAPTIVE TIERED HYBRID PIPELINE: FINAL PERFORMANCE")
    print("="*60)
    print(f"Cumulative Return (%)        : {(wealth.iloc[-1] - 1) * 100:.2f}%")
    print(f"Annualized Return (%)        : {ann_return * 100:.2f}%")
    print(f"Annualized Volatility (%)    : {ann_vol * 100:.2f}%")
    print(f"Sharpe Ratio                 : {sharpe:.2f}")
    print(f"Max Drawdown (%)             : {max_dd * 100:.2f}%")
    print(f"Average Daily Position Size  : {backtest_df['position_weight'].abs().mean():.4f}")
    print(f"Average Daily Turnover       : {backtest_df['turnover'].mean():.4f}")
    print("="*60)

    print("\n" + "="*60)
    print("🚀 ADAPTIVE TIERED HYBRID PIPELINE vs. BUY & HOLD BENCHMARK")
    print("="*60)
    print(f"{'Metric':<30} | {'Strategy':<15} | {'Buy & Hold':<15}")
    print("-" * 65)
    print(f"{'Cumulative Return (%)':<30} | {(wealth.iloc[-1] - 1) * 100:>14.2f}% | {bench_cum_return:>14.2f}%")
    print(f"{'Annualized Return (%)':<30} | {ann_return * 100:>14.2f}% | {bench_ann_return * 100:>14.2f}%")
    print(f"{'Annualized Volatility (%)':<30} | {ann_vol * 100:>14.2f}% | {bench_ann_vol * 100:>14.2f}%")
    print(f"{'Sharpe Ratio':<30} | {sharpe:>14.2f} | {bench_sharpe:>14.2f}")
    print(f"{'Max Drawdown (%)':<30} | {max_dd * 100:>14.2f}% | {bench_max_dd * 100:>14.2f}%")
    print("="*60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    backtest_df[[
        "date", "position_weight", "net_strategy_return", 
        "benchmark_return", "benchmark_wealth", "turnover", "rsi_14"
    ]].to_csv(output_path, index=False)

    print(f"💾 Adaptive tiered hybrid trade log saved to: {output_path}")

    return backtest_df

if __name__ == "__main__":
    run_backtest_pipeline()