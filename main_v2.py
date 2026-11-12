import argparse
import os
import sys
from dotenv import load_dotenv


#from src.data import run_full_data_pipeline
from src.data.merge_raw_collected import merge_raw_to_collected
from src.features import process_and_merge_csvs_v2
#from src.features.check_features import check_missing_features
from src.models.v2_alpha_engine import run_alpha_pipeline
from src.models.v2_volatility_engine import run_volat_pipeline


## Temporary usage
from src.data.binance_ingestion_engine import run_binance_parquet_ingestion_pipeline
from src.data.deribit_dvol_loader import fetch_and_store_daily_dvol_parquet

# 1. Load API keys from environment file
load_dotenv('token.env')
API_TOKEN = os.getenv('MY_TOKEN')


def main():
  # 2. Parse command-line flags
  parser = argparse.ArgumentParser(
      description='BTC Stacking Quantitative Trading System'
  )
  parser.add_argument(
      '--download-data',
      action='store_true',
      help='Fetch and update raw dataset from APIs',
  )
  parser.add_argument(
      '--merge-data',
      action='store_true',
      help='merge new fetched data to local collected data',
  )
  parser.add_argument(
      '--build-features',
      action='store_true',
      help='Processing downloaded data',
  )
  parser.add_argument(
      '--train',
      action='store_true',
      help='Train alpha and volatility engines',
  )
  parser.add_argument(
      "--backtest",
      action="store_true",
      help="Run vectorized backtest on holdout predictions",
  )
  parser.add_argument(
      "--test-ground",
      action="store_true",
      help="Train but holdout some dataset for backtest instead of OOF",
  )
#  parser.add_argument(
#      "--all",
#      action="store_true",
#      help="Run complete pipeline: download -> features -> train -> backtest",
#  )
#  parser.add_argument(
#      "--threshold",
#      type=float,
#      default=0.60,
#      help="Probability threshold for entering long position (default: 0.60)",
#  )

  args = parser.parse_args()

  # Handle --all flag shortcut
  #if args.all:
  #  args.download_data = True
  #  args.build_features = True
  #  args.train = True
  #  args.backtest = True

  #if not any([args.download_data, args.build_features, args.train, args.backtest]):
  #  parser.print_help()
  #  sys.exit(1)

  # 3. Route execution based on CLI flag
  if args.download_data:
  #  if not API_TOKEN:
  #    raise ValueError('API Token missing! Check your token.env file.')
  #  print('--> Triggering Data Ingestion Pipeline...')
  #  run_full_data_pipeline(api_token=API_TOKEN, symbol='BTCUSDT')
    run_binance_parquet_ingestion_pipeline()
    fetch_and_store_daily_dvol_parquet()
    
  if args.merge_data:
    print('--> Merging fetched data to collected data...')
    merge_raw_to_collected()

  if args.build_features:
    print('--> Building Features for Stacking Pipeline...')
    process_and_merge_csvs_v2()
    
  if args.train:
    print('--> Training Alpha and Volatility engines...')
  # run_alpha_pipeline()
    run_volat_pipeline()
   
  if args.backtest:
    print('--> Executing Backtesting Engine...')  
  #  #print(f"Executing Backtesting Engine (Threshold: {args.threshold})...")
  #  # args.threshold is passed directly here into the engine
    
  if args.test_ground:
    print('--> Training Alpha and Volatility engines...')
    
    
if __name__ == '__main__':
  main()