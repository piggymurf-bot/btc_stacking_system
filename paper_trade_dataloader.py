import os
import pandas as pd

# Import custom features 
# DataLoader
from src.data.bgeometrics_loader import fetch_and_merge_bgeometrics
from src.data.binance_vision import fetch_and_process_binance_vision
from src.data.ccxt_funding import fetch_and_process_funding_rates
from src.data.yfinance_dprice import fetch_yfinance_data

from dotenv import load_dotenv

# Load API key for BGeometrics data
load_dotenv('token.env')
API_TOKEN = os.getenv('MY_TOKEN')

if not API_TOKEN:
  raise ValueError(
      "MY_TOKEN environment variable is missing or empty! Check local token.env"
      " or GitHub Secrets."
  )

SYMBOL = "BTCUSDT"
TIMEFRAME = "1d"

# Directories for raw files
raw_dir: str = 'data/raw' 

def daily_dataloader(symbol=SYMBOL, timeframe=TIMEFRAME, limit=300):
    
    bg_df = fetch_and_merge_bgeometrics(API_TOKEN)

    # Extract date range from BGeometrics to synchronize all other data sources
    start_date = pd.to_datetime(bg_df['date'].iloc[0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(bg_df['date'].iloc[-1]).strftime('%Y-%m-%d')
    
    # Subtract 100 days so 50-day SMAs are fully calculated by 'start_date'
    yf_buffered_start = (pd.to_datetime(bg_df['date'].iloc[0]) - pd.Timedelta(days=100)).strftime(
      '%Y-%m-%d'
      )

    yf_buffered_end = (
            pd.to_datetime(bg_df['date'].iloc[-1]) + pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')

    fetch_and_process_binance_vision(symbol, start_date_str=start_date)

    fetch_and_process_funding_rates(start_date_str=start_date)

    fetch_yfinance_data(
        symbol='BTC-USD', start_date_str=yf_buffered_start, end_date_str=yf_buffered_end
    )
