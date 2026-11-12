import pandas as pd
import numpy as np

def run_bgeometrics_feature_extraction(bg_data: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw BGeometric On-Chain Data into stationary features for X.

    """
    print("⚡ Extracting BGeometrics Features from merged Dataset...")
    
    df = bg_data.copy()

    if(df.isna().values.any()):
        df = df.ffill()
        
    
    #Log-transform raw capital volumes to compress exponential variance spikes
    #We use a signed log transformation because nrplbtc contains negative numbers
    df['nrplbtc_log'] = np.sign(df['nrplbtc']) * np.log1p(df['nrplbtc'].abs())
    
    #Collect all safe features 
    safe_features = [
        'aviv', 'mvrv', 'nupl', 'sopr', 
    
        #Treated features
        'nrplbtc_log'
    ] 

    #Include also age cohorts 
    age_columns = [col for col in df.columns if col.startswith('age_')]
    safe_features.extend(age_columns)

    #Put all viable features into a new DataFrame
    features_df = df[safe_features]
    
    for col in features_df.columns:
        if col not in ["date", "timestamp"]:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
            
    #Add more features, which are the day-to-day difference and target variable

    #Create daily changes for key valuation metrics
    features_df['mvrv_daily_diff'] = features_df['mvrv'].diff(1)
    features_df['nupl_daily_diff'] = features_df['nupl'].diff(1)
    features_df['sopr_daily_diff'] = features_df['sopr'].diff(1)
    
    # Compute total circulating supply represented by all cohorts
    
    #Create daily changes for the supply cohorts
    age_columns = [col for col in features_df.columns if col.startswith('age_')]
    for col in age_columns:
        features_df[f'{col}_daily_diff'] = features_df[col].pct_change(1)

    # Transform age cohort from raw into share such that they are sumed to 1.0
    total_utxo_supply = df[age_columns].sum(axis=1)    
    # Convert each raw BTC cohort into a structural percentage share
    for col in age_columns:
        share_col_name = f"{col}_share"
        features_df[share_col_name] = df[col] / (total_utxo_supply + 1e-8)
    
        # Compute 1-day change on the percentage shares (Velocity / Migration)
        features_df[f"{share_col_name}_daily_diff"] = features_df[share_col_name].diff()
       
    # Base Macro Buckets (Levels)
    features_df['macro_share_sth_under_6m'] = (
        features_df.get('age_0d_1d_share', 0) + features_df.get('age_1d_1w_share', 0) + 
        features_df.get('age_1w_1m_share', 0) + features_df.get('age_1m_3m_share', 0) + 
        features_df.get('age_3m_6m_share', 0)
    )
    features_df['macro_share_mth_6m_1y'] = features_df.get('age_6m_1y_share', 0)
    features_df['macro_share_lth_1y_3y'] = features_df.get('age_1y_2y_share', 0) + features_df.get('age_2y_3y_share', 0)
    features_df['macro_share_dormant_3y_plus'] = (
        features_df.get('age_3y_4y_share', 0) + features_df.get('age_4y_5y_share', 0) + 
        features_df.get('age_5y_7y_share', 0) + features_df.get('age_7y_10y_share', 0) + 
        features_df.get('age_10y_plus_share', 0)
    )
    
    macro_buckets = [
        'macro_share_sth_under_6m', 
        'macro_share_mth_6m_1y', 
        'macro_share_lth_1y_3y', 
        'macro_share_dormant_3y_plus'
    ]    
    
    horizons = [1, 3, 7, 14]
    for bucket in macro_buckets:
       for h_diff in horizons:
           features_df[f'{bucket}_diff_{h_diff}d'] = features_df[bucket] - features_df[bucket].shift(h_diff)
       
    #Dropout the first row since its day-to-day difference will be NaN
    features_df = features_df.dropna()

    return features_df

if __name__ == "__main__":
    run_bgeometrics_feature_extraction()