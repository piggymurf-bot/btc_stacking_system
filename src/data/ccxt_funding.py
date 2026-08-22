import io
import os
import zipfile
import pandas as pd
import requests


def fetch_and_process_funding_rates(
    symbol: str = 'BTCUSDT',
    start_date_str: str = '2022-01-01',
    raw_dir: str = 'data/raw',
) -> pd.DataFrame:
  """Fetches historical funding rates directly from Binance Vision archives and aggregates to daily metrics."""

  # Format symbol for Binance Vision (e.g., BTCUSDT)
  clean_symbol = symbol.replace('/', '').split(':')[0]
  print(
      f'Fetching Binance Vision funding rates for {clean_symbol} starting from'
      f' {start_date_str}...'
  )

  # Generate monthly date range up to the current month
  start_dt = pd.to_datetime(start_date_str)
  end_dt = pd.Timestamp.now().floor('D')
  date_range = pd.date_range(start=start_dt, end=end_dt, freq='MS')

  all_dfs = []

  for dt in date_range:
    year_month = dt.strftime('%Y-%m')
    url = f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{clean_symbol}/{clean_symbol}-fundingRate-{year_month}.zip'

    try:
      response = requests.get(url, timeout=10)
      if response.status_code == 200:
        # Extract CSV in memory without writing the zip to disk
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
          csv_name = z.namelist()[0]
          with z.open(csv_name) as f:
            # Monthly files contain headers or raw columns
            df_month = pd.read_csv(f)
            # Standardize columns if unheadered
            if 'calc_time' not in df_month.columns:
              df_month.columns = [
                  'calc_time',
                  'funding_interval_hours',
                  'last_funding_rate',
              ]
            all_dfs.append(df_month)
      else:
        print(
            f'Archive not found or current month for {year_month} (Status'
            f' {response.status_code})'
        )
    except Exception as e:
      print(f'Failed to download {year_month}: {e}')

  if not all_dfs:
    print('No Binance Vision funding rate archives retrieved.')
    return pd.DataFrame()

  # Combine and clean data
  df_funding = pd.concat(all_dfs, ignore_index=True)

  # Map column names
  time_col = (
      'calc_time' if 'calc_time' in df_funding.columns else df_funding.columns[0]
  )
  rate_col = (
      'last_funding_rate'
      if 'last_funding_rate' in df_funding.columns
      else df_funding.columns[2]
  )

  df_funding['datetime'] = pd.to_datetime(
      df_funding[time_col], unit='ms', utc=True
  )
  df_funding['fundingRate'] = df_funding[rate_col].astype(float)
  df_funding['date'] = df_funding['datetime'].dt.floor('D')

  # Aggregate to daily metrics
  df_daily_funding = (
      df_funding.groupby('date')
      .agg(
          funding_rate_daily_sum=('fundingRate', 'sum'),
          funding_rate_daily_mean=('fundingRate', 'mean'),
          funding_rate_daily_max=('fundingRate', 'max'),
      )
      .reset_index()
  )

  df_daily_funding['date'] = df_daily_funding['date'].dt.tz_localize(None)

  os.makedirs(raw_dir, exist_ok=True)
  output_path = os.path.join(raw_dir, 'daily_funding.csv')
  df_daily_funding.to_csv(output_path, index=False)

  print(
      f'Saved aggregated funding rates into {len(df_daily_funding)} daily'
      f' records at {output_path}'
  )

  return df_daily_funding