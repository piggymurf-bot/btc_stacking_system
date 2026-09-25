import pandas as pd
import os

def merged_csv(old_data_path: str, new_data_path: str) -> pd.DataFrame:
    # ---------------------------------------------------------
    # 1. READ OR LOAD YOUR TWO BGEOMETRICS DATAFRAMES
    # ---------------------------------------------------------
  
    df_older = pd.read_csv(old_data_path, index_col=0, parse_dates=True)
    df_newer = pd.read_csv(new_data_path, index_col=0, parse_dates=True)

    # Ensure datetime index and timezone-naive
    df_older.index = pd.to_datetime(df_older.index).tz_localize(None)
    df_newer.index = pd.to_datetime(df_newer.index).tz_localize(None)

    # ---------------------------------------------------------
    # 2. COMBINE AND DEDUPLICATE
    # ---------------------------------------------------------
    # Option A: Stack vertically and keep the 'newer' dataset's values for overlapping dates
    merged_df = pd.concat([df_older, df_newer])

    # Remove duplicate date index rows, keeping the last occurrence (from df_newer)
    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]

    # Sort chronologically from oldest to newest date
    merged_df.sort_index(inplace=True)

    # ---------------------------------------------------------
    # 3. CLEAN UP ANY GAPS OR MISSING COLUMNS
    # ---------------------------------------------------------
    # Forward-fill any minor missing values across the merged timeline
    #bgeometrics_df.ffill(inplace=True)

    print(
        f"Merged Dataset range: {merged_df.index.min().strftime('%Y-%m-%d')} "
        f"to {merged_df.index.max().strftime('%Y-%m-%d')}"
    )
    print(f'Total Daily Rows: {len(merged_df)}')
    return merged_df


def merge_raw_to_collected(raw_dir: str = 'data/raw'):
    
    collected_files = ['bgeometrics_collected.csv',
                       #'binancevision_collected.csv',
                       #'dailyfunding_collected.csv',
                       #'yfinance_collected.csv'
                       ]
    
    raw_files = ['bgeometrics_merged.csv',
                 #'binance_vision_metrics.csv',
                 #'daily_funding.csv',
                 #'yfinance_data.csv'
                 ]
    
    for raw_file, collected_file in zip(raw_files, collected_files):
        
        old_path = os.path.join(raw_dir, collected_file)
        new_path = os.path.join(raw_dir, raw_file)
        
        merged_df = merged_csv(old_path, new_path)
        merged_df.to_csv(old_path)
        
    print("Successfully merge raw files to collected files")
    