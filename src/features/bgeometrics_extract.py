import pandas as pd
import numpy as np


def extract_bgeometrics_features(bg_data: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw BGeometric On-Chain Data into stationary features for X.

    """
    df = bg_data.copy()

    if(df.isna().values.any()):
        df = df.ffill()
        
    #Treating features that are broken by the standard scaler
    
    #Convert absolute MACD values into a Percentage Price Oscillator (PPO) 
    #by scaling them by the long-term trend baseline (ema200)
    df['macd_pct'] = df['macd'] / df['ema200']
    df['macdsignal_pct'] = df['macdsignal'] / df['ema200']
    df['macdhist_pct'] = df['macdhist'] / df['ema200']

    #Log-transform raw capital volumes to compress exponential variance spikes
    #We use a signed log transformation because nrplbtc contains negative numbers
    df['nrplbtc_log'] = np.sign(df['nrplbtc']) * np.log1p(df['nrplbtc'].abs())

    #Collect all safe features 
    safe_features = [
        'aviv', 'mvrv', 'nupl', 'sopr', 
    
        #Treated features
        'macd_pct', 'macdsignal_pct', 'macdhist_pct', 'nrplbtc_log'
    ]

    #Include also age cohorts 
    age_columns = [col for col in df.columns if col.startswith('age_')]
    safe_features.extend(age_columns)

    #Put all viable features into a new DataFrame
    features_df = df[safe_features]

    #Add more features, which are the day-to-day difference and target variable

    #Create daily changes for key valuation metrics
    features_df['mvrv_daily_diff'] = features_df['mvrv'].diff(1)
    features_df['nupl_daily_diff'] = features_df['nupl'].diff(1)
    
    #Create daily changes for the supply cohorts
    age_columns = [col for col in features_df.columns if col.startswith('age_')]
    for col in age_columns:
        features_df[f'{col}_daily_diff'] = features_df[col].pct_change(1)

    #Dropout the first row since its day-to-day difference will be NaN
    features_df = features_df.dropna()

    return(features_df)