from datetime import datetime, timedelta, timezone
import io
import os
import zipfile
import pandas as pd
import requests

BASE_URL = 'https://data.binance.vision/data/futures/um/daily/metrics'


def fetch_binance_vision_metrics(
    symbol: str, start_date_str: str, end_date_str: str
) -> pd.DataFrame:
  """Downloads daily futures metrics ZIP files directly from Binance Data Vision."""
  start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
  end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

  curr_date = start_date
  df_list = []

  print(
      f'Fetching Binance Vision metrics for {symbol} from {start_date_str} to'
      f' {end_date_str}...'
  )

  while curr_date <= end_date:
    date_str = curr_date.strftime('%Y-%m-%d')
    file_name = f'{symbol}-metrics-{date_str}.zip'
    url = f'{BASE_URL}/{symbol}/{file_name}'

    response = requests.get(url)
    if response.status_code == 200:
      try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
          csv_filename = z.namelist()[0]
          with z.open(csv_filename) as f:
            daily_df = pd.read_csv(f)
            df_list.append(daily_df)
      except Exception as e:
        print(f'Error unzipping {date_str}: {e}')

    curr_date += timedelta(days=1)

  if not df_list:
    raise ValueError('No Binance Vision data was downloaded.')

  return pd.concat(df_list, ignore_index=True)


def process_binance_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
  """Aggregates intraday metrics to daily and generates stationary features."""
  df = raw_df.copy()
  df.columns = [col.strip().lower() for col in df.columns]

  date_col = 'create_time' if 'create_time' in df.columns else 'timestamp'
  df['datetime'] = pd.to_datetime(df[date_col])
  df['date'] = df['datetime'].dt.floor('D')

  aggregated_df = (
      df.groupby('date')
      .agg(
          open_interest_usd=('sum_open_interest_value', 'last'),
          long_short_ratio=('count_long_short_ratio', 'last'),
          taker_buy_sell_ratio=('sum_taker_long_short_vol_ratio', 'last'),
      )
      .reset_index()
  )

  aggregated_df['date'] = pd.to_datetime(aggregated_df['date']).dt.tz_localize(
      None
  )

  # Calculate stationary features
  aggregated_df['oi_pct_change_1d'] = aggregated_df[
      'open_interest_usd'
  ].pct_change()
  aggregated_df['oi_pct_change_7d'] = aggregated_df[
      'open_interest_usd'
  ].pct_change(7)

  ls_mean = (
      aggregated_df['long_short_ratio'].rolling(14, min_periods=1).mean()
  )
  ls_std = aggregated_df['long_short_ratio'].rolling(14, min_periods=1).std()
  aggregated_df['ls_ratio_zscore_14d'] = (
      aggregated_df['long_short_ratio'] - ls_mean
  ) / (ls_std + 1e-8)

  return aggregated_df


def fetch_and_process_binance_vision(
    symbol: str,
    start_date_str: str,
    end_date_str: str = None,
    raw_dir: str = 'data/raw',
) -> pd.DataFrame:
  """Main wrapper function for Binance Vision extraction."""
  if end_date_str is None:
    end_date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        '%Y-%m-%d'
    )

  raw_df = fetch_binance_vision_metrics(symbol, start_date_str, end_date_str)
  processed_df = process_binance_metrics(raw_df)

  output_path = os.path.join(raw_dir, 'binance_vision_metrics.csv')
  processed_df.to_csv(output_path, index=False)
  print(f'Saved Binance Vision metrics to {output_path}')

  return processed_df