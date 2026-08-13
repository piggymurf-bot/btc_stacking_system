import pandas as pd


def extract_binancevision_features(binancev_data: pd.DataFrame) -> pd.DataFrame:
    """Remove some non-stationary features from raw Binance Vision, leaving only stationary features for X.

    """
    df = binancev_data.copy()

    #Collect only stationary features
    safe_features = [
        'oi_pct_change_1d' ,'ls_ratio_zscore_14d'
    ]

    #Put all viable features into a new DataFrame
    features_df = df[safe_features]
    
    return(features_df)