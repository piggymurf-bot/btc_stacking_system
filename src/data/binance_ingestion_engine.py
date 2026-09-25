from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import io
import os
import zipfile
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

BASE_URL = "https://data.binance.vision/data/futures/um"
RAW_PARQUET_DIR = "data/raw/parquet"
PROCESSED_DIR = "data/processed"
MAX_WORKERS = 16

def get_configured_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS * 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# =====================================================================
# Download & Store Raw Parquet Files
# =====================================================================

def _download_and_save_kline_parquet(date_str: str, symbol: str, session: requests.Session):
    """Downloads raw 1m Klines ZIP and saves compressed Parquet file to disk."""
    out_dir = os.path.join(RAW_PARQUET_DIR, "klines_1m")
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"{symbol}-1m-{date_str}.parquet")

    if os.path.exists(parquet_path):
        return date_str

    url = f"{BASE_URL}/daily/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip"
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5, 8, 9])
                    if str(df.iloc[0, 0]).lower() == 'open_time':
                        df = df.iloc[1:].reset_index(drop=True)
                    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_vol']
                    df = df.astype(float)
                    df['date_str'] = date_str
                    df.to_parquet(parquet_path, compression='snappy', index=False)
                    return date_str
    except Exception:
        pass
    return None


def _download_and_save_depth_parquet(date_str: str, symbol: str, session: requests.Session):
    """Downloads raw L2 bookDepth ZIP and saves compressed Parquet file to disk."""
    out_dir = os.path.join(RAW_PARQUET_DIR, "book_depth")
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"{symbol}-bookDepth-{date_str}.parquet")

    if os.path.exists(parquet_path):
        return date_str

    url = f"{BASE_URL}/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    df.columns = [col.lower().strip() for col in df.columns]
                    df['date_str'] = date_str
                    df.to_parquet(parquet_path, compression='snappy', index=False)
                    return date_str
    except Exception:
        pass
    return None


def _download_and_save_metrics_parquet(date_str: str, symbol: str, session: requests.Session):
    """Downloads daily metrics ZIP (OI, LS Ratio) and saves compressed Parquet file to disk."""
    out_dir = os.path.join(RAW_PARQUET_DIR, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"{symbol}-metrics-{date_str}.parquet")

    if os.path.exists(parquet_path):
        return date_str

    url = f"{BASE_URL}/daily/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    df.columns = [col.strip().lower() for col in df.columns]
                    df['date_str'] = date_str
                    df.to_parquet(parquet_path, compression='snappy', index=False)
                    return date_str
    except Exception:
        pass
    return None


def _download_and_save_funding_parquet(symbol: str, start_date_str: str, session: requests.Session):
    """Downloads monthly funding rate archives and saves raw Parquet files to disk."""
    out_dir = os.path.join(RAW_PARQUET_DIR, "funding_rates")
    os.makedirs(out_dir, exist_ok=True)

    date_range = pd.date_range(start=start_date_str, end=pd.Timestamp.now().floor('D'), freq='MS')

    for dt in date_range:
        year_month = dt.strftime('%Y-%m')
        parquet_path = os.path.join(out_dir, f"{symbol}-fundingRate-{year_month}.parquet")

        if os.path.exists(parquet_path):
            continue

        url = f"{BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year_month}.zip"
        try:
            res = session.get(url, timeout=10)
            if res.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        df_month = pd.read_csv(f)
                        if 'calc_time' not in df_month.columns:
                            df_month.columns = ['calc_time', 'funding_interval_hours', 'last_funding_rate']
                        df_month.to_parquet(parquet_path, compression='snappy', index=False)
        except Exception:
            pass


def run_binance_parquet_ingestion_pipeline(
    symbol: str = "BTCUSDT",
    start_date_str: str = "2020-01-01",
    end_date_str: str = None
):
    if end_date_str is None:
        end_date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    date_range = [dt.strftime("%Y-%m-%d") for dt in pd.date_range(start_date_str, end_date_str, freq="D")]
    session = get_configured_session()

    print(f"📡 Downloading & Caching Raw Parquet Files ({MAX_WORKERS} threads)...")

    # Concurrent downloads for daily endpoints
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures_kline = [executor.submit(_download_and_save_kline_parquet, d, symbol, session) for d in date_range]
        futures_depth = [executor.submit(_download_and_save_depth_parquet, d, symbol, session) for d in date_range if d >= "2023-01-01"]
        futures_metrics = [executor.submit(_download_and_save_metrics_parquet, d, symbol, session) for d in date_range if d >= "2020-01-01"]

        for _ in as_completed(futures_kline + futures_depth + futures_metrics):
            pass

    # Monthly funding rates download
    _download_and_save_funding_parquet(symbol, start_date_str, session)

    return None

if __name__ == "__main__":
    run_binance_parquet_ingestion_pipeline(start_date_str="2020-01-01")