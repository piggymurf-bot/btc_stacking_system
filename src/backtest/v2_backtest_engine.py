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


def audit_retrain_windows(bt_df, retrain_mask_col="retrained", lookahead=10, lookback=3):
    """
    Inspects bars around retraining events to check for sign inversion between
    pred_alpha / position and subsequent market returns.

    If you don't have a boolean `retrained` column in `bt_df`, it falls back to
    detecting sharp PSI drops after a breach above 0.25.
    """
    df = bt_df.copy()

    # Identify event indices where a retrain reset occurred
    if retrain_mask_col in df.columns:
        event_positions = np.where(df[retrain_mask_col].astype(bool).values)[0]
    else:
        psi = df["feature_psi"].values
        event_positions = []
        for idx in range(1, len(psi)):
            # Detect a drop back near baseline right after crossing > 0.25
            if psi[idx - 1] > 0.25 and psi[idx] < psi[idx - 1] * 0.5:
                event_positions.append(idx)

    print(f"Found {len(event_positions)} candidate retrain/reset events.\n")

    cols_to_show = [
        c for c in [
            "market_regime",
            "feature_psi",
            "pred_alpha",
            "effective_alpha",
            "position",
            "actual_return",
            "strategy_net_return"
        ] if c in df.columns
    ]

    for k, pos in enumerate(event_positions[:5], start=1):  # inspect first 5 events
        start_idx = max(0, pos - lookback)
        end_idx = min(len(df), pos + lookahead + 1)
        window = df.iloc[start_idx:end_idx][cols_to_show].copy()

        # Flag days where prediction sign disagrees with realized market return
        if "pred_alpha" in window.columns and "actual_return" in window.columns:
            window["sign_match"] = np.sign(window["pred_alpha"]) == np.sign(window["actual_return"])

        print("=" * 78)
        print(f"RETRAIN EVENT #{k} around index {pos} (date={df.index[pos]})")
        print("=" * 78)
        print(window.to_string())
        print()


def run_volatility_scaled_backtest(
    backtest_csv_path: str = "data/backtest/holdout_backtest.csv",
    target_annual_vol: float = 0.25,
    max_leverage: float = 1.0,
    fee_bps: float = 5.0,
    alpha_threshold: float = 0.002,
    alpha_scale: float = 0.03,         # Alpha magnitude that reaches full 1.0 conviction
    use_rsi_filter: bool = True,
    rsi_overbought: float = 75.0,
    rsi_oversold: float = 25.0,
    drawdown_window: int = 14,
    drawdown_veto_threshold: float = -0.07,
    min_pred_vol: float = 0.05,
    use_gk_dampener: bool = False      # Enable Garman-Klass volatility spike dampener
):
    bt_df = pd.read_csv(backtest_csv_path, parse_dates=['date'], index_col='date')

    # 1. Underlying Daily Asset Returns
    bt_df['actual_return'] = bt_df['close'].pct_change(fill_method=None)

    # 2. Continuous Alpha Conviction Scaling With Deadband Around 0
    # Forecasts inside [-alpha_threshold, alpha_threshold] stay flat (0.0);
    # stronger forecasts scale smoothly up to [-1.0, 1.0] at +/- alpha_scale.
    raw_conviction = np.clip(bt_df['pred_alpha'] / alpha_scale, -1.0, 1.0)
    bt_df['raw_signal'] = np.where(
        bt_df['pred_alpha'].abs() > alpha_threshold,
        raw_conviction,
        0.0
    )

    # 3. Optional RSI Safety Gate
    if use_rsi_filter and 'rsi_14' in bt_df.columns:
        bt_df['raw_signal'] = np.where(
            (bt_df['raw_signal'] > 0.0) & (bt_df['rsi_14'] > rsi_overbought),
            0.0,
            bt_df['raw_signal']
        )
        bt_df['raw_signal'] = np.where(
            (bt_df['raw_signal'] < 0.0) & (bt_df['rsi_14'] < rsi_oversold),
            0.0,
            bt_df['raw_signal']
        )

    # 4. Trailing Drawdown Circuit Breaker (Crash Protection)
    rolling_max_close = bt_df['close'].rolling(window=drawdown_window, min_periods=1).max()
    market_drawdown_from_peak = (bt_df['close'] - rolling_max_close) / rolling_max_close
    bt_df['market_dd'] = market_drawdown_from_peak

    # Veto long signals if price drops past threshold from its rolling peak
    bt_df['raw_signal'] = np.where(
        (market_drawdown_from_peak < drawdown_veto_threshold) & (bt_df['raw_signal'] > 0.0),
        0.0,
        bt_df['raw_signal']
    )

    # 5. Garman-Klass Volatility Circuit Breaker (Hard-Kill Flat Override)
    if use_gk_dampener and 'garman_klass_vol' in bt_df.columns:
        # Compute rolling median baseline for intraday range volatility
        baseline_gk_vol = bt_df['garman_klass_vol'].rolling(window=7, min_periods=1).median()

        # Calculate volatility expansion ratio
        gk_vol_ratio = bt_df['garman_klass_vol'] / (baseline_gk_vol + 1e-8)

        # If intraday range volatility blows out past 1.5x normal, force signal straight to flat (0.0)
        bt_df['raw_signal'] = np.where(gk_vol_ratio > 1.5, 0.0, bt_df['raw_signal'])

    # 6. Volatility Targeting / Risk Parity Sizing & Leverage Clipping
    if 'pred_volat' in bt_df.columns:
        safe_pred_vol = bt_df['pred_volat'].fillna(min_pred_vol).clip(lower=min_pred_vol)
    else:
        safe_pred_vol = pd.Series(min_pred_vol, index=bt_df.index)

    vol_scalar = target_annual_vol / safe_pred_vol
    bt_df['target_position'] = (bt_df['raw_signal'] * vol_scalar).clip(
        upper=max_leverage,
        lower=-max_leverage
    )

    # 7. Shift position by 1 bar to prevent lookahead bias
    bt_df['position'] = bt_df['target_position'].shift(1).fillna(0.0)

    # Diagnostic column: alpha forecast from bar t-1 that produced today's position
    bt_df['effective_alpha'] = bt_df['pred_alpha'].shift(1)

    # 8. Turnover & Transaction Costs
    bt_df['turnover'] = bt_df['position'].diff().abs().fillna(0.0)
    cost_per_unit = fee_bps / 10000.0
    bt_df['transaction_cost'] = bt_df['turnover'] * cost_per_unit

    # 9. Strategy Returns
    bt_df['strategy_gross_return'] = bt_df['position'] * bt_df['actual_return']
    bt_df['strategy_net_return'] = bt_df['strategy_gross_return'] - bt_df['transaction_cost']

    # 10. Cumulative Equity Curves
    bt_df['cumulative_market'] = (1 + bt_df['actual_return'].fillna(0.0)).cumprod()
    bt_df['cumulative_strategy'] = (1 + bt_df['strategy_net_return'].fillna(0.0)).cumprod()

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

    bt_df['feature_psi'] = bt_df['psi']

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

    audit_retrain_windows(bt_df, lookahead=10, lookback=2)

    return bt_df


if __name__ == '__main__':
    results_df = run_volatility_scaled_backtest()