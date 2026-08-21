import os
import pandas as pd
import yfinance as yf


def fetch_yfinance_data(
    symbol: str = 'BTC-USD',
    start_date_str: str = '2022-01-01',
    end_date_str: str = None,
    raw_dir: str = 'data/raw',
) -> pd.DataFrame:
  """Downloads daily OHLCV price data from Yahoo Finance."""
  print(
      f'Fetching YFinance spot data for {symbol} ({start_date_str} to'
      f' {end_date_str})...'
  )

  btc_data = yf.download(
      symbol, start=start_date_str, end=end_date_str, progress=False, repair=True
  )

  # Flatten MultiIndex columns if yfinance returns nested headers
  if isinstance(btc_data.columns, pd.MultiIndex):
    btc_data.columns = btc_data.columns.get_level_values(0)

  btc_data.reset_index(inplace=True)
  btc_data.rename(columns={'Date': 'date'}, inplace=True)
  btc_data['date'] = pd.to_datetime(btc_data['date']).dt.tz_localize(None)

  os.makedirs(raw_dir, exist_ok=True)
  output_path = os.path.join(raw_dir, 'yfinance_data.csv')
  btc_data.to_csv(output_path, index=False)
  print(f'Saved YFinance data ({len(btc_data)} rows) to {output_path}')

  return btc_data