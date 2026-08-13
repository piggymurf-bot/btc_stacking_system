import numpy as np
import pandas as pd


def extract_yfinance_features(btc_data: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw yfinance OHLCV DataFrame into stationary features for X.

    Expects columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    df = btc_data.copy()

    features = pd.DataFrame(index=df.index)

    # 1. Volatility Features
    # High-Low Range %
    features['yf_hl_spread_pct'] = (df['High'] - df['Low']) / df['Close']

    # True Range & ATR 14
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs(),
        ),
    )
    atr14 = tr.rolling(14).mean()
    features['yf_atr14_pct'] = atr14 / df['Close']  # Scale-independent ATR

    # Bollinger Band %B
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    features['yf_bollinger_pct_b'] = (df['Close'] - lower_bb) / (
        upper_bb - lower_bb + 1e-8
    )

    # 2. Trend & Momentum Ratios
    # SMA Ratios & Distances
    sma7 = df['Close'].rolling(7).mean()
    sma50 = df['Close'].rolling(50).mean()
    features['yf_sma7_to_sma50_ratio'] = sma7 / (sma50 + 1e-8)
    features['yf_dist_from_sma20'] = (df['Close'] - sma20) / (sma20 + 1e-8)

    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['yf_rsi14'] = 100 - (100 / (1 + rs))

    # 3. Volume Metrics
    vol_sma20 = df['Volume'].rolling(20).mean()
    features['yf_volume_ratio_20d'] = df['Volume'] / (vol_sma20 + 1e-8)

    # 4. Stationary Returns
    features['yf_return_1d'] = df['Close'].pct_change(1)
    features['yf_return_7d'] = df['Close'].pct_change(7)
        
    # 5. Keeping 'Close': the spot closing price for the backtesting 
    features['Close'] = df['Close']

    # Drop initial NaN rows caused by rolling windows
    features = features.dropna()
    
    return features