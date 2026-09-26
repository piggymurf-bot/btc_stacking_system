import os
import numpy as np
import pandas as pd

# Annualization factor for daily crypto data (365 trading days/year)
ANN_FACTOR = np.sqrt(365)
EPSILON = 1e-8  # Small constant to prevent divide-by-zero errors

def compute_forward_alpha_targets(
    df: pd.DataFrame,
    horizons: list = [1, 3, 7, 14],
    close_col: str = "close",
    price_return_col: str = None
) -> pd.DataFrame:
    """
    Computes forward log returns, forward realized volatility, and forward annualized Sharpe ratios.
    
    Target Calculation:
        y_{t, H} = [ Forward_Mean_Return(t, t+H) * 365 ] / [ Forward_Std_Dev(t, t+H) * sqrt(365) + eps ]
    """
    df = df.copy()
    
    # 1. Compute 1-day log returns if not present
    if price_return_col and price_return_col in df.columns:
        ret_1d = df[price_return_col]
    else:
        ret_1d = np.log(df[close_col] / df[close_col].shift(1))
        df["log_ret_1d"] = ret_1d

    # Shift log returns by -1 so row 't' contains the return realised over t -> t+1
    fwd_ret_1d = ret_1d.shift(-1)

    # 2. Compute multi-horizon targets
    for h in horizons:
        # Forward cumulative log return over horizon H: sum of future 1d returns
        fwd_cum_ret = fwd_ret_1d.iloc[::-1].rolling(window=h, min_periods=h).sum().iloc[::-1]
        
        # Forward mean daily return over horizon H
        fwd_mean_ret = fwd_ret_1d.iloc[::-1].rolling(window=h, min_periods=h).mean().iloc[::-1]
        
        # Ensure min_periods does not exceed the window size h
        min_p = min(h, max(2, h // 2)) if h > 1 else 1
        
        # Forward daily volatility over horizon H (sample std dev)
        fwd_vol_daily = fwd_ret_1d.iloc[::-1].rolling(window=h, min_periods=min_p).std().iloc[::-1]
        
        # Annualized values
        fwd_vol_ann = fwd_vol_daily * ANN_FACTOR
        fwd_mean_ann = fwd_mean_ret * 365.0
        
        # Forward Annualized Sharpe Ratio (handle h=1 gracefully where vol is NaN)
        fwd_sharpe = fwd_mean_ann / (fwd_vol_ann + EPSILON) if h > 1 else np.nan

        # Store engineered target columns
        df[f'target_fwd_cum_ret_{h}d'] = fwd_cum_ret
        df[f'target_fwd_vol_ann_{h}d'] = fwd_vol_ann
        df[f'target_fwd_sharpe_{h}d'] = fwd_sharpe
    
    # 3. Interplay parameters between volatility and alpha signal
    df["iv_rv_spread"] = (df["dvol_close"] / 100) - df["rv_1m_daily"]
    df["dvol_rv_ratio"] = (df["dvol_close"] / 100)/( df["rv_1m_daily"] + 1e-8)
    
    return df


def process_alpha_targets(
    dir_path: str = "data/processed",
    matrix_file: str = "master_feature_addlhar.csv",
    #horizons: list = [1, 3, 7, 14]
    horizons: list = [1]
):
    """Loads raw data, constructs alpha targets, drops trailing NaNs, and saves the output."""
    master_path = os.path.join(dir_path, matrix_file)
    
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Input data not found at {master_path}. Run data ingestion first.")

    print(f"📥 Loading raw dataset from: {master_path}")
    df = pd.read_csv(master_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    print(f"⚙️ Computing forward Sharpe targets for horizons: {horizons} days...")
    df_with_targets = compute_forward_alpha_targets(df, horizons=horizons)

    # Max shift horizon will result in trailing NaNs at the end of time series
    max_h = max(horizons)
    df_clean = df_with_targets.iloc[:-max_h].copy()

    os.makedirs(os.path.dirname(dir_path), exist_ok=True)
    df_path = os.path.join(dir_path, 'master_feature_addlhar_alpha.csv')
    df_clean.to_csv(df_path, index=False)
    
    print(f"✅ Generated {len(df_clean)} rows of alpha target matrix.")
    print(f"📁 Processed dataset saved to: {df_path}")

    return df_clean


if __name__ == "__main__":
    process_alpha_targets()