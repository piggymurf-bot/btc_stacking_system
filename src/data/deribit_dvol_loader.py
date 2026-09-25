import os
import time
import requests
import pandas as pd

DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
RAW_PARQUET_DIR = "data/raw/parquet/deribit_dvol"
PROCESSED_DIR = "data/processed"

def fetch_and_store_daily_dvol_parquet(
    currency: str = "BTC",
    start_date_str: str = "2021-03-25"
) -> str:
    """Paginates 1D DVOL candles backward to fetch full history from 2021 to present."""
    os.makedirs(RAW_PARQUET_DIR, exist_ok=True)
    parquet_path = os.path.join(RAW_PARQUET_DIR, f"deribit_dvol_{currency.lower()}_1d_raw.parquet")
    
    start_ts = int(pd.Timestamp(start_date_str, tz="UTC").timestamp() * 1000)
    current_end_ts = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    
    print(f"📡 Fetching Complete Historical Deribit DVOL ({currency}, Daily resolution) via Pagination...")
    
    seen_timestamps = set()
    all_records = []

    while current_end_ts > start_ts:
        params = {
            "currency": currency,
            "start_timestamp": start_ts,
            "end_timestamp": current_end_ts,
            "resolution": "1D"
        }
        
        try:
            res = requests.get(DERIBIT_URL, params=params, timeout=15).json()
            records = res.get("result", {}).get("data", [])
            
            if not records:
                break

            # Filter out duplicate candles across batch boundaries
            new_records = [r for r in records if r[0] not in seen_timestamps]
            if not new_records:
                break
                
            for r in new_records:
                seen_timestamps.add(r[0])
            all_records.extend(new_records)

            # Move current_end_ts past the oldest timestamp in this batch
            batch_min_ts = min(r[0] for r in records)
            current_end_ts = batch_min_ts - 1
            time.sleep(0.02)

        except Exception as e:
            print(f"❌ Error during pagination: {e}")
            break

    if not all_records:
        print("❌ No records fetched from Deribit API.")
        return ""

    df_raw = pd.DataFrame(all_records, columns=["timestamp", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])
    df_raw = df_raw.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    df_raw["date"] = df_raw["datetime"].dt.floor("D").dt.tz_localize(None)
    
    df_raw.to_parquet(parquet_path, compression="snappy", index=False)
    print(f"✅ Full Raw DVOL Parquet saved to: {parquet_path} ({len(df_raw)} total daily rows)")
    return parquet_path

if __name__ == "__main__":
    fetch_and_store_daily_dvol_parquet()