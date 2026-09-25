import os
import numpy as np
import pandas as pd

RAW_PARQUET_DIR = "data/raw/parquet/deribit_dvol"
PROCESSED_DIR = "data/processed"

def extract_dvol_daily_features(parquet_path: str) -> pd.DataFrame:
    """Reads raw daily DVOL Parquet and extracts stationary volatility features."""
    if not os.path.exists(parquet_path):
        return pd.DataFrame()

    print("⚡ Extracting daily DVOL features from local Parquet datalake...")
    df_daily = pd.read_parquet(parquet_path).sort_values("date").reset_index(drop=True)

    # Intraday Range
    df_daily["dvol_intraday_range"] = (df_daily["dvol_high"] - df_daily["dvol_low"]) / (df_daily["dvol_low"] + 1e-8)
    
    # Parkinson Range Estimator
    log_hl_sq = (np.log(df_daily["dvol_high"] / df_daily["dvol_low"])) ** 2
    df_daily["dvol_parkinson"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl_sq)

    # Stationary derivatives
    df_daily["dvol_daily_diff"] = df_daily["dvol_close"].diff(1)
    df_daily["dvol_pct_change_1d"] = df_daily["dvol_close"].pct_change(1)
    
    # 14-day DVOL Z-score
    rolling_mean = df_daily["dvol_close"].rolling(14, min_periods=1).mean()
    rolling_std = df_daily["dvol_close"].rolling(14, min_periods=1).std()
    df_daily["dvol_zscore_14d"] = (df_daily["dvol_close"] - rolling_mean) / (rolling_std + 1e-8)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "deribit_dvol_features.csv")
    df_daily.to_csv(out_path, index=False)
    
    print(f"✅ Processed Deribit DVOL features saved to: {out_path} ({len(df_daily)} rows)")
    return df_daily

def run_deribit_feature_extraction(
    raw_dir: str = RAW_PARQUET_DIR,
    processed_dir: str = PROCESSED_DIR     
) -> pd.DataFrame:
    
    print("⚡ Extracting Deribit features from Parquet file...")

    deribit_dir = os.path.join(RAW_PARQUET_DIR, "deribit_dvol_btc_1d_raw.parquet")
    df_features = extract_dvol_daily_features(deribit_dir)
        
    return df_features

if __name__ == "__main__":
    run_deribit_feature_extraction()