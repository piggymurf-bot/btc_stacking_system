import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_volatility_scaled_backtest(
    backtest_csv_path: str = "data/backtest/holdout_backtest.csv",
    target_annual_vol: float = 0.50,  # Target 50% annualized portfolio volatility
    max_leverage: float = 1.0,        # Cap maximum leverage (e.g., 1.0 = no margin)
    fee_bps: float = 5.0              # 5 basis points per unit of turnover (fees + slippage)
):
    """
    Loads the holdout backtest dataset, computes volatility-scaled position sizes,
    deducts transaction costs, and compares net performance against Bitcoin buy-and-hold.
    """
    bt_df = pd.read_csv(backtest_csv_path, parse_dates=['date'], index_col='date')
    
    # 1. Underlying Daily Asset Returns
    bt_df['actual_return'] = bt_df['close'].pct_change(fill_method=None)
    
    # 2. Base Directional Signal from Alpha Prediction (+1 for Long, 0 for Flat)
    # You can also use -1 if you want to allow shorting: np.where(bt_df['pred_alpha'] > 0, 1.0, -1.0)
    bt_df['raw_signal'] = np.where(bt_df['pred_alpha'] > 0, 1.0, 0.0)
    
    # 3. Volatility Targeting / Risk Parity Sizing
    # Position scales inversely to predicted volatility: higher predicted vol -> smaller size
    # Avoid division by zero by clipping pred_volat at a small positive floor (e.g., 0.05)
    safe_pred_vol = bt_df['pred_volat'].clip(lower=0.05)
    vol_scalar = target_annual_vol / safe_pred_vol
    
    # Combine directional signal with volatility scaling and clip leverage
    bt_df['target_position'] = bt_df['raw_signal'] * vol_scalar
    bt_df['target_position'] = bt_df['target_position'].clip(upper=max_leverage, lower=0.0) # Long-only constraint
    
    # 4. Shift position by 1 bar to prevent lookahead bias (executed on the next close)
    bt_df['position'] = bt_df['target_position'].shift(1).fillna(0)
    
    # 5. Turnover & Transaction Costs
    bt_df['turnover'] = bt_df['position'].diff().abs().fillna(0)
    cost_per_unit = fee_bps / 10000.0
    bt_df['transaction_cost'] = bt_df['turnover'] * cost_per_unit
    
    # 6. Strategy Returns (Gross vs Net)
    bt_df['strategy_gross_return'] = bt_df['position'] * bt_df['actual_return']
    bt_df['strategy_net_return'] = bt_df['strategy_gross_return'] - bt_df['transaction_cost']
    
    # 7. Cumulative Equity Curves ($1.00 Starting Base)
    bt_df['cumulative_market'] = (1 + bt_df['actual_return'].fillna(0)).cumprod()
    bt_df['cumulative_strategy'] = (1 + bt_df['strategy_net_return'].fillna(0)).cumprod()
    
    # 8. Performance Metrics Calculation
    total_days = len(bt_df)
    ann_factor = np.sqrt(365) # Crypto annualization factor
    
    net_sharpe = (bt_df['strategy_net_return'].mean() / (bt_df['strategy_net_return'].std() + 1e-8)) * ann_factor
    total_net_return = bt_df['cumulative_strategy'].iloc[-1] - 1
    total_market_return = bt_df['cumulative_market'].iloc[-1] - 1
    
    rolling_max = bt_df['cumulative_strategy'].cummax()
    drawdown = (bt_df['cumulative_strategy'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    total_net_net_str = f"{total_net_return * 100:.2f}%"
    
    print("=" * 60)
    print("📈 VOLATILITY-SCALED BACKTEST PERFORMANCE REPORT (NET)")
    print("=" * 60)
    print(f"Holdout Duration:             {total_days} bars")
    print(f"Strategy Cumulative Net Return: {total_net_net_str} ({total_net_return * 100:.2f}%)")    
    print(f"Benchmark (BTC Buy & Hold):     {total_market_return * 100:.2f}%")
    print(f"Annualized Net Sharpe Ratio:    {net_sharpe:.2f}")
    print(f"Maximum Drawdown:             {max_drawdown * 100:.2f}%")
    print(f"Average Daily Turnover:       {bt_df['turnover'].mean() * 100:.2f}%")
    print("=" * 60)
    
    # 9. Plotting Results
    plt.figure(figsize=(11, 5))
    plt.plot(bt_df.index, bt_df['cumulative_strategy'], label='Vol-Scaled Regime Strategy (Net)', color='dodgerblue', linewidth=2)
    plt.plot(bt_df.index, bt_df['cumulative_market'], label='Bitcoin Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    plt.title('Holdout PnL: Volatility-Scaled Stacking Strategy vs. BTC Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($1.00 Base)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return bt_df

if __name__ == '__main__':
    results_df = run_volatility_scaled_backtest()