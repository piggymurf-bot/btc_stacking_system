import os
import ccxt
import pandas as pd


def fetch_and_process_funding_rates(
    symbol: str = 'BTC/USDT:USDT',
    start_date_str: str = '2022-01-01',
    raw_dir: str = 'data/raw',
) -> pd.DataFrame:
  """Fetches historical funding rates via CCXT and aggregates to daily metrics."""
  
  exchange = ccxt.bitget({'enableRateLimit': True})
  #exchange = ccxt.binanceusdm({'enableRateLimit': True})
  #exchange = ccxt.bybit({'enableRateLimit': True})
  since = exchange.parse8601(f'{start_date_str}T00:00:00Z')
  now_ms = exchange.milliseconds()

  all_funding_records = []
  print(f'Fetching CCXT funding rates for {symbol} from {start_date_str}...')

  while True:
    rates = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
    if not rates:
      break

    all_funding_records.extend(rates)
    last_timestamp = rates[-1]['timestamp']

    if last_timestamp <= since or last_timestamp > now_ms:
      break
    since = last_timestamp + 1

  df_funding = pd.DataFrame(all_funding_records)
  df_funding['datetime'] = pd.to_datetime(
      df_funding['timestamp'], unit='ms', utc=True
  )
  df_funding['date'] = df_funding['datetime'].dt.floor('D')

  # Aggregate 8-hour intervals to daily
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
      f'Saved {len(all_funding_records)} funding rates aggregated into'
      f' {len(df_daily_funding)} daily records at {output_path}'
  )

  return df_daily_funding