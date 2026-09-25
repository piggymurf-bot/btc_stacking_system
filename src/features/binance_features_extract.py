import os
import numpy as np
import pandas as pd

RAW_PARQUET_DIR = "data/raw/parquet"
PROCESSED_DIR = "data/processed"

# =====================================================================
# Local Feature Extraction
# =====================================================================

def extract_daily_kline_features(kline_parquet_path: str) -> dict:
    """Reads raw 1m Parquet file for A SINGLE DAY and extracts intraday metrics."""
    df = pd.read_parquet(kline_parquet_path)
    date_str = df['date_str'].iloc[0]

    d_open = df['open'].iloc[0]
    d_high = df['high'].max()
    d_low = df['low'].min()
    d_close = df['close'].iloc[-1]    
    d_volume = df['volume'].sum()
    
    # Intraday VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    d_vwap = np.sum(typical_price * df['volume']) / (d_volume + 1e-8)
        
    # Intraday 1-minute Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    returns = df['log_return'].dropna().values

    # Realized Volatilities (Intraday)
    rv_daily = np.sqrt(np.sum(returns ** 2))
    rv_pos = np.sqrt(np.sum(returns[returns > 0] ** 2)) if np.any(returns > 0) else 0.0
    rv_neg = np.sqrt(np.sum(returns[returns < 0] ** 2)) if np.any(returns < 0) else 0.0

    # Intraday Range Volatility Estimators
    log_hl_sq = (np.log(df['high'] / df['low'])) ** 2
    parkinson_vol = np.sqrt((1.0 / (4.0 * np.log(2.0))) * np.sum(log_hl_sq))

    log_co_sq = (np.log(df['close'] / df['open'])) ** 2
    gk_elem = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    garman_klass_vol = np.sqrt(np.maximum(0.0, np.sum(gk_elem)))

    # Return pure daily metrics (No multi-day rolling features here!)
    return {
        'date': pd.to_datetime(date_str),
        'open': d_open,
        'high': d_high,
        'low': d_low,
        'close': d_close,
        'volume': d_volume,
        'trades': df['trades'].sum(),
        'vwap': d_vwap,
        'taker_buy_ratio': df['taker_buy_vol'].sum() / (d_volume + 1e-8),
        'rv_1m_daily': rv_daily,
        'rv_pos_1m': rv_pos,
        'rv_neg_1m': rv_neg,
        'parkinson_vol': parkinson_vol,
        'garman_klass_vol': garman_klass_vol
    }

def compute_multi_day_technicals(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Computes multi-day rolling indicators across the stacked daily (technicals) DataFrame."""
    df_daily = df_daily.sort_values("date").reset_index(drop=True)
    
    close = df_daily["close"]
    high = df_daily["high"]
    low = df_daily["low"]
    volume = df_daily["volume"]

    # 1. Log Returns & Spreads
    df_daily["close_log_return_1d"] = np.log(close / close.shift(1))
    df_daily["close_log_return_7d"] = np.log(close / close.shift(7))
    df_daily["hl_spread_pct"] = (high - low) / close

    # 2. ATR 14 (Normalized)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    df_daily["atr_14_pct"] = atr14 / close

    # 3. Volume Ratio & Moving Averages
    df_daily["volume_ratio_20d"] = volume / (volume.rolling(20, min_periods=1).mean() + 1e-8)
    sma7 = close.rolling(7, min_periods=1).mean()
    sma20 = close.rolling(20, min_periods=1).mean()
    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()

    # 4. Distance Ratios
    df_daily["sma7_to_sma50_ratio"] = sma7 / (sma50 + 1e-8)
    df_daily["dist_from_sma20"] = (close - sma20) / (sma20 + 1e-8)
    df_daily["dist_from_sma200"] = (close - sma200) / (sma200 + 1e-8)

    # 5. Bollinger %B
    std20 = close.rolling(20, min_periods=1).std()
    upper_b = sma20 + (2 * std20)
    lower_b = sma20 - (2 * std20)
    df_daily["bollinger_pct_b"] = (close - lower_b) / (upper_b - lower_b + 1e-8)

    # 6. RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df_daily["rsi_14"] = 100 - (100 / (1 + rs))

    # 7. MACD Histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    #df_daily["macd_hist"] = macd_line - macd_signal
    df_daily["macd_hist_pct"] = (macd_line - macd_signal) / close

    # 8. vwap_dist_pct
    df_daily["vwap_dist_pct"] = (close - df_daily["vwap"]) / df_daily["vwap"]

    return df_daily

def extract_daily_depth_features(depth_parquet_path: str) -> dict:
    """Reads raw bookDepth Parquet file from disk and aggregates L2 features."""
    df = pd.read_parquet(depth_parquet_path)
    date_str = df['date_str'].iloc[0]

    pivoted = df.pivot(index='timestamp', columns='percentage', values='depth')
    b1 = pivoted.get(-1, pd.Series(0, index=pivoted.index))
    a1 = pivoted.get(1, pd.Series(0, index=pivoted.index))

    obi_1m = (b1 - a1) / (b1 + a1 + 1e-8)
    total_depth_1m = b1 + a1

    return {
        'date': pd.to_datetime(date_str),
        'obi_mean_daily': obi_1m.mean(),
        'obi_std_daily': obi_1m.std(),
        'book_depth_1m_mean': total_depth_1m.mean(),
        'bid_ask_depth_ratio': (b1.mean() / (a1.mean() + 1e-8))
    }

def compute_multi_day_depth(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Computes multi-day rolling indicators across the stacked daily (book depth) DataFrame."""
    df_daily = df_daily.sort_values("date").reset_index(drop=True)
    
    book_depth_1m_mean = df_daily["book_depth_1m_mean"]
    sma_30_book_depth_1m_mean = book_depth_1m_mean.rolling(30, min_periods=1).mean()
    depth_ratio_30d = book_depth_1m_mean/(sma_30_book_depth_1m_mean + 1e-8)
    df_daily['depth_ratio_30d'] = depth_ratio_30d
    
    return df_daily
    

def extract_daily_metrics_features(metrics_dir: str) -> pd.DataFrame:
    """Reads all raw metrics Parquet files and computes daily stationary metrics."""
    metrics_files = sorted([os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith(".parquet")])
    records = []

    for f in metrics_files:
        df = pd.read_parquet(f)
        date_str = df['date_str'].iloc[0]
        
        records.append({
            'date': pd.to_datetime(date_str),
            'open_interest_usd': df['sum_open_interest_value'].iloc[-1],
            'long_short_ratio': df['count_long_short_ratio'].iloc[-1],
            'taker_buy_sell_ratio': df['sum_taker_long_short_vol_ratio'].iloc[-1]
        })

    if not records:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(records).sort_values('date').reset_index(drop=True)

    # Stationary derivatives
    df_metrics['oi_pct_change_1d'] = df_metrics['open_interest_usd'].pct_change()
    df_metrics['oi_pct_change_7d'] = df_metrics['open_interest_usd'].pct_change(7)

    ls_mean = df_metrics['long_short_ratio'].rolling(14, min_periods=1).mean()
    ls_std = df_metrics['long_short_ratio'].rolling(14, min_periods=1).std()
    df_metrics['ls_ratio_zscore_14d'] = (df_metrics['long_short_ratio'] - ls_mean) / (ls_std + 1e-8)

    return df_metrics


def extract_daily_funding_features(funding_dir: str) -> pd.DataFrame:
    """Reads raw monthly funding Parquet files and aggregates daily metrics."""
    funding_files = sorted([os.path.join(funding_dir, f) for f in os.listdir(funding_dir) if f.endswith(".parquet")])
    all_dfs = [pd.read_parquet(f) for f in funding_files]

    if not all_dfs:
        return pd.DataFrame()

    df_funding = pd.concat(all_dfs, ignore_index=True)
    time_col = 'calc_time' if 'calc_time' in df_funding.columns else df_funding.columns[0]
    rate_col = 'last_funding_rate' if 'last_funding_rate' in df_funding.columns else df_funding.columns[2]

    df_funding['datetime'] = pd.to_datetime(df_funding[time_col], unit='ms', utc=True)
    df_funding['fundingRate'] = df_funding[rate_col].astype(float)
    df_funding['date'] = df_funding['datetime'].dt.floor('D').dt.tz_localize(None)

    df_daily_funding = df_funding.groupby('date').agg(
        funding_rate_daily_sum=('fundingRate', 'sum'),
        funding_rate_daily_mean=('fundingRate', 'mean'),
        funding_rate_daily_max=('fundingRate', 'max')
    ).reset_index()

    return df_daily_funding

def run_binance_feature_extraction(
    raw_dir: str = RAW_PARQUET_DIR,
    processed_dir: str = PROCESSED_DIR     
) -> pd.DataFrame:
    
    print("⚡ Extracting Binance Vision Features from Parquet Datalake...")

    kline_dir = os.path.join(RAW_PARQUET_DIR, "klines_1m")
    depth_dir = os.path.join(RAW_PARQUET_DIR, "book_depth")
    metrics_dir = os.path.join(RAW_PARQUET_DIR, "metrics")
    funding_dir = os.path.join(RAW_PARQUET_DIR, "funding_rates")

    kline_files = sorted([os.path.join(kline_dir, f) for f in os.listdir(kline_dir) if f.endswith(".parquet")])
    depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".parquet")])

    kline_features = [extract_daily_kline_features(f) for f in kline_files]
    depth_features = [extract_daily_depth_features(f) for f in depth_files]

    df_klines = pd.DataFrame(kline_features).sort_values('date')
    df_depth = pd.DataFrame(depth_features).sort_values('date') if depth_features else pd.DataFrame()
    df_metrics = extract_daily_metrics_features(metrics_dir) if os.path.exists(metrics_dir) else pd.DataFrame()
    df_funding = extract_daily_funding_features(funding_dir) if os.path.exists(funding_dir) else pd.DataFrame()

    # Add multi-day technicals
    df_klines = compute_multi_day_technicals(df_klines)
    df_depth = compute_multi_day_depth(df_depth)

    # Merge all feature streams
    df_merged = df_klines.merge(df_metrics, on="date", how="left") if not df_metrics.empty else df_klines
    if not df_funding.empty:
        df_merged = df_merged.merge(df_funding, on="date", how="left")
    if not df_depth.empty:
        df_merged = df_merged.merge(df_depth, on="date", how="left")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "binance_unified_features.csv")
    df_merged.to_csv(out_path, index=False)

    print(f"✅ Local Parquet cache populated in: {RAW_PARQUET_DIR}")
    print(f"📁 Processed feature matrix ({df_merged.shape[0]} rows x {df_merged.shape[1]} cols) saved to: {out_path}")

    return df_merged    

if __name__ == "__main__":

  run_binance_feature_extraction()      

  