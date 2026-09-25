import os
import pandas as pd
import numpy as np

from src.features.bgeometrics_extract import extract_bgeometrics_features
from src.features.binancevision_extract import extract_binancevision_features
from src.features.yfinance_extract import extract_yfinance_features
from src.features.target_builder import add_target_variable
from src.features.fracdiff import frac_diff_ffd,find_optimal_d

#######

from src.features.bgeometrics_features_extract import run_bgeometrics_feature_extraction
from src.features.binance_features_extract import run_binance_feature_extraction
from src.features.deribit_features_extract import run_deribit_feature_extraction
from src.features.build_lhar_features import add_lhar_main
from src.features.build_alpha_targets import process_alpha_targets
from src.features.check_features import check_missing_features

def process_and_merge_csvs(raw_dir: str = 'data/raw', out_dir: str = 'data/processed') -> dict[str, pd.DataFrame]:
    """Loads all four raw CSV files, extract needed features and merge them based on date index/columns."""

    # Processing BGeometrics data
    file_path = os.path.join(raw_dir, 'bgeometrics_merged.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
        
    bgeometrics_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    bgeometrics_data_processed = extract_bgeometrics_features(bgeometrics_data)
    
    # Processing Binance Vision data
    file_path = os.path.join(raw_dir, 'binance_vision_metrics.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
    
    binancevision_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    binancevision_data_processed = extract_binancevision_features(binancevision_data)
    
    # Processing Funding Rate data
    file_path = os.path.join(raw_dir, 'daily_funding.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
    
    dailyfund_data_processed = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    # Processing Yahoo Finance data
    file_path = os.path.join(raw_dir, 'yfinance_data.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
      
    yfinance_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    yfinance_data_processed = extract_yfinance_features(yfinance_data)


    # Begin by putting BGeometrics data on the master DataFrame
    master_df = bgeometrics_data_processed.copy()

    # Sequential merge on date
    master_df = pd.merge(master_df, binancevision_data_processed, on='date', how='inner')
    master_df = pd.merge(master_df, dailyfund_data_processed, on='date', how='inner')
    master_df = pd.merge(master_df, yfinance_data_processed, on='date', how='inner')
      
    # List of cumulative non-stationary features to transform
    non_stationary_cols = ['Close', 'mvrv', 'aviv', 'nrplbtc_log']
    
    # Assign the DataFrame to collect optimum d
    d_opt_df = pd.DataFrame(columns=non_stationary_cols,index=range(1))

    for col in non_stationary_cols:
        print(f'Finding optimal d* for {col}...')

        # 1. Take log of price/index if not already logged (stabilizes variance)
        series_to_diff = (
            np.log(master_df[col]) if col == 'Close' or (master_df[col] > 0).all() else master_df[col]
            )

        # 2. Find minimum d* that passes ADF test (stationarity)
        optimal_d = find_optimal_d(series_to_diff, p_thres=0.05)

        # 3. Apply FFD using optimal d*
        master_df[f'{col}_fracdiff'] = frac_diff_ffd(series_to_diff, d=optimal_d)
        
        # 4. Collect the optimal d* for each column
        d_opt_df[f'{col}'] = optimal_d

    # Attach Target Variable y (Predicting 3-day forward SMA movement)
    master_df = add_target_variable(
          master_df, close_col='Close', horizon_days=3
      )
    
    # Drop initial NaNs created by the FFD window (typically ~30 to 50 rows)
    #master_df = master_df.dropna().reset_index(drop=False)
    master_df = master_df.dropna()
    
    # Save merged dataset
    master_path = os.path.join(out_dir, 'master_features_and_target.csv')
    master_df.to_csv(master_path, index=True)
    
    # Save optimal d for processing daily dataset 
    d_opt_path = os.path.join(out_dir, 'optimal_d.csv')
    d_opt_df.to_csv(d_opt_path, index=True)
    
    print('\n=== DOWNLOADED DATA SUCCESSFULLY PROCESSED & SAVED TO data/processed/ ===')
    print(f'Master Dataset Shape: {master_df.shape[0]} rows x {master_df.shape[1]} columns')
    
    target_dist = master_df['target'].value_counts(normalize=True) * 100
    print(f'Target Distribution (y):\n{target_dist.round(2).to_string()}')

    return(master_df)
    
def process_and_merge_csvs_v2(raw_dir: str = 'data/raw', raw_parquet_dir: str = "data/raw/parquet",  out_dir: str = 'data/processed') -> dict[str, pd.DataFrame]:
    """Process raw files; begometrics CSV and Binanace/Deribit PARQUET, extract needed features and merge them based on date index/columns."""
    
    # Processing BGeometrics data
    #file_path = os.path.join(raw_dir, 'bgeometrics_merged.csv')
    file_path = os.path.join(raw_dir, 'bgeometrics_collected.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
    bgeometrics_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    bgeometrics_data_processed = run_bgeometrics_feature_extraction(bgeometrics_data)
    binance_data_processed = run_binance_feature_extraction(raw_parquet_dir,out_dir) 
    deribit_processed = run_deribit_feature_extraction(raw_parquet_dir,out_dir)
   
    # Begin by putting BGeometrics data on the master DataFrame
    master_df = bgeometrics_data_processed.copy()

    # Sequential merge on date
    master_df = pd.merge(master_df, binance_data_processed, on='date', how='inner')
    master_df = pd.merge(master_df, deribit_processed, on='date', how='inner')
    
    # Dropping multiple columns at once
    master_df = master_df.drop(columns=['timestamp', 'datetime'])
    
    # List of cumulative non-stationary features to transform
    non_stationary_cols = [ 'aviv', 'mvrv', 'nupl', 'nrplbtc_log', 'close']
    
    # Assign the DataFrame to collect optimum d
    d_opt_df = pd.DataFrame(columns=non_stationary_cols,index=range(1))

    for col in non_stationary_cols:
        print(f'Finding optimal d* for {col}...')

        # 1. Take log of price/index if not already logged (stabilizes variance)
        series_to_diff = (
            np.log(master_df[col]) if col == 'close' or (master_df[col] > 0).all() else master_df[col]
            )

        # 2. Find minimum d* that passes ADF test (stationarity)
        optimal_d = find_optimal_d(series_to_diff, p_thres=0.05)

        # 3. Apply FFD using optimal d*
        master_df[f'{col}_fracdiff'] = frac_diff_ffd(series_to_diff, d=optimal_d)
        
        # 4. Collect the optimal d* for each column
        d_opt_df[f'{col}'] = optimal_d
    
    # Save merged dataset
    master_path = os.path.join(out_dir, 'master_feature_matrix.csv')
    master_df.to_csv(master_path, index=True)
    
    # Save optimal d for processing daily dataset 
    d_opt_path = os.path.join(out_dir, 'optimal_d_v2.csv')
    d_opt_df.to_csv(d_opt_path, index=True)
    
    # Add volatility parameters/target 
    add_lhar_main()
    # Add alpha target/ alpha-volatility paramters
    process_alpha_targets()
    
    # Looking if there are missing features from the Feature Groups used in training 
    check_missing_features()
    
    
    
    
    