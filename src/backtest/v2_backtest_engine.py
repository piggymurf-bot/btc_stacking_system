import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_psi(reference_series, current_series, num_bins=10):
    """Calculates Population Stability Index between reference and current feature distributions."""
    # Compute bin edges based on reference distribution quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(reference_series.dropna(), quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Get counts
    ref_counts = np.histogram(reference_series.dropna(), bins=bins)[0]
    curr_counts = np.histogram(current_series.dropna(), bins=bins)[0]
    
    # Convert to percentages (with small epsilon to avoid division by zero)
    ref_pct = (ref_counts / len(reference_series.dropna())) + 1e-4
    curr_pct = (curr_counts / len(current_series.dropna())) + 1e-4
    
    # Re-normalize percentages
    ref_pct /= ref_pct.sum()
    curr_pct /= curr_pct.sum()
    
    # PSI formula
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return psi

def run_volatility_scaled_backtest(
    backtest_csv_path: str = "data/backtest/holdout_backtest.csv",
    target_annual_vol: float = 0.50,  
    max_leverage: float = 1.0,        
    fee_bps: float = 5.0,             
    alpha_threshold: float = 0.003,   
    use_rsi_filter: bool = True,      
    use_gk_dampener: bool = True      # Enable Garman-Klass volatility spike dampener
):
    bt_df = pd.read_csv(backtest_csv_path, parse_dates=['date'], index_col='date')
    
    # 1. Underlying Daily Asset Returns
    bt_df['actual_return'] = bt_df['close'].pct_change(fill_method=None)
    
    # 2. Center Alpha Predictions via Rolling Mean
    rolling_mean_alpha = bt_df['pred_alpha'].rolling(window=14, min_periods=1).mean()
    bt_df['centered_alpha'] = bt_df['pred_alpha'] - rolling_mean_alpha
    
    # 3. Confidence Threshold / Deadband
    bt_df['raw_signal'] = np.where(
        bt_df['centered_alpha'] > alpha_threshold, 1.0,
        np.where(bt_df['centered_alpha'] < -alpha_threshold, -1.0, 0.0)
    )
    
    # 4. Optional RSI Safety Gate
    if use_rsi_filter and 'rsi_14' in bt_df.columns:
        bt_df['raw_signal'] = np.where((bt_df['raw_signal'] == 1.0) & (bt_df['rsi_14'] > 75), 0.0, bt_df['raw_signal'])
        bt_df['raw_signal'] = np.where((bt_df['raw_signal'] == -1.0) & (bt_df['rsi_14'] < 25), 0.0, bt_df['raw_signal'])

    ## Trailing Drawdown Circuit Breaker (Crash Protection)
    # Track the running peak of the preliminary target positions or apply a portfolio stop.
    # A clean way is to compute a running peak of the underlying price or equity, 
    # but let's implement a direct rolling drawdown override on the signal:
    
    # Calculate rolling peak of close price to detect local market tops/waterfalls
    rolling_max_close = bt_df['close'].rolling(window=20, min_periods=1).max()
    market_drawdown_from_peak = (bt_df['close'] - rolling_max_close) / rolling_max_close
    
    # If the market drops more than 7% rapidly from its 20-day peak, veto all long signals (force flat or short)
    bt_df['raw_signal'] = np.where(
        (market_drawdown_from_peak < -0.07) & (bt_df['raw_signal'] == 1.0), 
        0.0, 
        bt_df['raw_signal']
    )

    # 5. Volatility Targeting / Risk Parity Sizing
    safe_pred_vol = bt_df['pred_volat'].clip(lower=0.05)
    vol_scalar = target_annual_vol / safe_pred_vol
    
    bt_df['target_position'] = bt_df['raw_signal'] * vol_scalar

   # =========================================================================
    # 6. Garman-Klass Volatility Circuit Breaker (Hard-Kill Flat Override)
    # =========================================================================
    if use_gk_dampener and 'garman_klass_vol' in bt_df.columns:
        # Compute 30-day rolling median baseline for intraday range volatility
        baseline_gk_vol = bt_df['garman_klass_vol'].rolling(window=7, min_periods=1).median()
        
        # Calculate volatility expansion ratio
        gk_vol_ratio = bt_df['garman_klass_vol'] / (baseline_gk_vol + 1e-8)
        
        # If intraday range volatility blows out past 1.5x normal, force signal straight to flat (0.0)
        bt_df['raw_signal'] = np.where(gk_vol_ratio > 1.5, 0.0, bt_df['raw_signal'])
    # =========================================================================

    bt_df['target_position'] = bt_df['target_position'].clip(upper=max_leverage, lower=-max_leverage)
    
    # 7. Shift position by 1 bar to prevent lookahead bias
    bt_df['position'] = bt_df['target_position'].shift(1).fillna(0)
    
    # 8. Turnover & Transaction Costs
    bt_df['turnover'] = bt_df['position'].diff().abs().fillna(0)
    cost_per_unit = fee_bps / 10000.0
    bt_df['transaction_cost'] = bt_df['turnover'] * cost_per_unit
    
    # 9. Strategy Returns
    bt_df['strategy_gross_return'] = bt_df['position'] * bt_df['actual_return']
    bt_df['strategy_net_return'] = bt_df['strategy_gross_return'] - bt_df['transaction_cost']
    
    # 10. Cumulative Equity Curves
    bt_df['cumulative_market'] = (1 + bt_df['actual_return'].fillna(0)).cumprod()
    bt_df['cumulative_strategy'] = (1 + bt_df['strategy_net_return'].fillna(0)).cumprod()
    
    # 11. Performance Metrics Calculation
    total_days = len(bt_df)
    ann_factor = np.sqrt(365)
    
    net_sharpe = (bt_df['strategy_net_return'].mean() / (bt_df['strategy_net_return'].std() + 1e-8)) * ann_factor
    total_net_return = bt_df['cumulative_strategy'].iloc[-1] - 1
    total_market_return = bt_df['cumulative_market'].iloc[-1] - 1
    
    rolling_max = bt_df['cumulative_strategy'].cummax()
    drawdown = (bt_df['cumulative_strategy'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    print("=" * 60)
    print("📈 VOLATILITY-SCALED BACKTEST (GK DAMPENER ACTIVE)")
    print("=" * 60)
    print(f"Holdout Duration:              {total_days} bars")
    print(f"Strategy Cumulative Net Return: {total_net_return * 100:.2f}%")    
    print(f"Benchmark (BTC Buy & Hold):      {total_market_return * 100:.2f}%")
    print(f"Annualized Net Sharpe Ratio:     {net_sharpe:.2f}")
    print(f"Maximum Drawdown:              {max_drawdown * 100:.2f}%")
    print(f"Average Daily Turnover:        {bt_df['turnover'].mean() * 100:.2f}%")
    print(f"Long / Short / Flat Breakdown: {(bt_df['position'] > 0).sum()} Long / {(bt_df['position'] < 0).sum()} Short / {(bt_df['position'] == 0).sum()} Flat")
    print("=" * 60)
    
    # Establish reference baseline from the first 30 days of holdout
    ref_gk = bt_df['garman_klass_vol'].iloc[:30]
    
    # Compute rolling PSI for Garman-Klass volatility over time
    psi_list = []
    window_size = 30
    for i in range(len(bt_df)):
        if i < window_size:
            psi_list.append(0.0)
        else:
            curr_window = bt_df['garman_klass_vol'].iloc[i - window_size:i]
            psi_list.append(calculate_psi(ref_gk, curr_window))
    bt_df['feature_psi'] = psi_list

    # Compute rolling 14-day out-of-sample Sharpe ratio
    rolling_strat_ret = bt_df['strategy_net_return']
    bt_df['rolling_sharpe_14d'] = (rolling_strat_ret.rolling(14).mean() / (rolling_strat_ret.rolling(14).std() + 1e-8)) * np.sqrt(365)
    
    # 12. Four-Panel Forensic Audit Plotting
    fig, axes = plt.subplots(
        nrows=4, 
        ncols=1, 
        figsize=(12, 11), 
        sharex=True, 
        gridspec_kw={'height_ratios': [3, 1, 1.2, 1.2]}
    )
    
    # Panel 1: PnL Equity Curve
    axes[0].plot(bt_df.index, bt_df['cumulative_strategy'], label='GK-Dampened Strategy (Net)', color='dodgerblue', linewidth=2)
    axes[0].plot(bt_df.index, bt_df['cumulative_market'], label='Bitcoin Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    axes[0].set_title('Holdout PnL, Position Audit & Regime Drift Diagnostics')
    axes[0].set_ylabel('Portfolio Value ($1.00)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Daily Position Size
    axes[1].fill_between(bt_df.index, bt_df['position'], 0, where=(bt_df['position'] > 0), color='forestgreen', alpha=0.6, label='Long')
    axes[1].fill_between(bt_df.index, bt_df['position'], 0, where=(bt_df['position'] < 0), color='crimson', alpha=0.6, label='Short')
    axes[1].fill_between(bt_df.index, bt_df['position'], 0, where=(bt_df['position'] == 0), color='gray', alpha=0.3, label='Flat')
    axes[1].set_ylabel('Position')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend(loc='upper left', fontsize=8, ncol=3)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Population Stability Index (Data Drift Detector)
    axes[2].plot(bt_df.index, bt_df['feature_psi'], color='darkorange', linewidth=1.8, label='GK Vol PSI (30d Rolling)')
    axes[2].axhline(0.10, color='gold', linestyle='--', linewidth=1, label='Moderate Drift (0.10)')
    axes[2].axhline(0.25, color='red', linestyle='--', linewidth=1.2, label='Severe Drift / Retrain Trigger (0.25)')
    axes[2].fill_between(bt_df.index, bt_df['feature_psi'], 0.25, where=(bt_df['feature_psi'] >= 0.25), color='red', alpha=0.2)
    axes[2].set_ylabel('PSI Score')
    axes[2].legend(loc='upper left', fontsize=8, ncol=3)
    axes[2].grid(True, alpha=0.3)
    
    # Panel 4: Rolling 14-Day Out-of-Sample Sharpe Ratio (Performance Degradation)
    axes[3].plot(bt_df.index, bt_df['rolling_sharpe_14d'].fillna(0), color='purple', linewidth=1.8, label='14d Rolling Sharpe')
    axes[3].axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.6)
    axes[3].fill_between(bt_df.index, bt_df['rolling_sharpe_14d'].fillna(0), 0, where=(bt_df['rolling_sharpe_14d'] < 0), color='crimson', alpha=0.25, label='Negative Sharpe Zone')
    axes[3].set_ylabel('14d Sharpe')
    axes[3].set_xlabel('Date')
    axes[3].legend(loc='upper left', fontsize=8, ncol=2)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return bt_df

if __name__ == '__main__':
    results_df = run_volatility_scaled_backtest()