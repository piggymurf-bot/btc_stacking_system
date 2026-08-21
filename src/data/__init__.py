import pandas as pd
from src.data.bgeometrics_loader import fetch_and_merge_bgeometrics
from src.data.binance_vision import fetch_and_process_binance_vision
from src.data.ccxt_funding import fetch_and_process_funding_rates
from src.data.yfinance_dprice import fetch_yfinance_data


def run_full_data_pipeline(api_token: str, symbol: str = 'BTCUSDT'):
  """Orchestrates all four data source downloads using BGeometrics date ranges."""
  # Step 1: Fetch BGeometrics
  bg_df = fetch_and_merge_bgeometrics(api_token)

  # Extract date range from BGeometrics to synchronize all other data sources
  start_date = pd.to_datetime(bg_df['date'].iloc[0]).strftime('%Y-%m-%d')
  end_date = pd.to_datetime(bg_df['date'].iloc[-1]).strftime('%Y-%m-%d')
  
  # Subtract 100 days so 50-day SMAs are fully calculated by 'start_date'
  yf_buffered_start = (pd.to_datetime(bg_df['date'].iloc[0]) - pd.Timedelta(days=100)).strftime(
    '%Y-%m-%d'
    )
  #Add 1 more day for the Yahoo Finance dataset.. to get the latest data on the same day
  yf_buffered_end = (
          pd.to_datetime(bg_df['date'].iloc[-1]) + pd.Timedelta(days=1)
      ).strftime('%Y-%m-%d')

  # Step 2: Fetch Binance Vision
  fetch_and_process_binance_vision(symbol, start_date_str=start_date)

  # Step 3: Fetch CCXT Funding Rates
  fetch_and_process_funding_rates(start_date_str=start_date)

  # Step 4: Fetch YFinance Price Action
  fetch_yfinance_data(
      symbol='BTC-USD', start_date_str=yf_buffered_start, end_date_str=yf_buffered_end
  )

  print('\n=== ALL DATA SOURCES SUCCESSFULLY DOWNLOADED & SAVED TO data/raw/ ===')