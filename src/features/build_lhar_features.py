import os
import numpy as np
import pandas as pd


def build_lhar_features(
    df: pd.DataFrame, 
    rv_col: str = "rv_1m_daily",
    rv_pos_col: str = "rv_pos_1m",
    rv_neg_col: str = "rv_neg_1m"
) -> pd.DataFrame:
    """Calculates Leveraged HAR-RV (LHAR-RV) components:

    - Standard HAR lags: RV_d, RV_w, RV_m
    - Asymmetric Leverage components: RV_neg_d, RV_pos_d, RV_neg_w, RV_pos_w
    """
    df = df.copy()
    
    # 1. Target variable: Next day's Realized Volatility (RV_t+1)
    df["target_rv"] = df[rv_col].shift(-1)
    
    # 2. Traditional HAR Components
    df["rv_d"] = df[rv_col]                             # Daily (1d)
    df["rv_w"] = df[rv_col].rolling(window=7).mean()    # Weekly (7d)
    df["rv_m"] = df[rv_col].rolling(window=30).mean()   # Monthly (30d)
    
    # 3. Asymmetric Downside / Upside Semivariance Lags
    if rv_neg_col in df.columns and rv_pos_col in df.columns:
        # Daily Downside vs Upside Realized Volatility
        df["rv_neg_d"] = df[rv_neg_col]
        df["rv_pos_d"] = df[rv_pos_col]
        
        # Weekly Downside vs Upside Realized Volatility
        df["rv_neg_w"] = df[rv_neg_col].rolling(window=7).mean()
        df["rv_pos_w"] = df[rv_pos_col].rolling(window=7).mean()
             
    else:
        # Fallback approximation if semivariance columns are not explicitly pre-computed
        print("⚠️ Warning: Pre-computed semivariance columns missing. Falling back to return direction splitting.")
        if "close" in df.columns:
            daily_returns = df["close"].pct_change()
            df["rv_neg_d"] = np.where(daily_returns < 0, df[rv_col], 0.0)
            df["rv_pos_d"] = np.where(daily_returns >= 0, df[rv_col], 0.0)
            df["rv_neg_w"] = df["rv_neg_d"].rolling(window=7).mean()
            df["rv_pos_w"] = df["rv_pos_d"].rolling(window=7).mean()

    # Daily Asymmetry Ratio (1-day view)
    df["rv_asymmetry_ratio_d"] = np.log((df["rv_pos_d"] + 1e-8) / (df["rv_neg_d"] + 1e-8))
    # Weekly Asymmetry Ratio (7-day rolling view - smoother macro trend)
    df["rv_asymmetry_ratio_w"] = np.log((df["rv_pos_w"] + 1e-8) / (df["rv_neg_w"] + 1e-8))

    return df

def add_lhar_main(
        dir_path: str = "data/processed",
        matrix_file: str = "master_feature_matrix.csv",
) -> pd.DataFrame:
    master_path = os.path.join(dir_path, matrix_file)
    
    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path)
        df = df_master.copy()
        
        # Build LHAR asymmetric features
        df = build_lhar_features(
            df, 
            rv_col="rv_1m_daily", 
            rv_pos_col="rv_pos_1m", 
            rv_neg_col="rv_neg_1m"
        )
        df_path = os.path.join(dir_path, 'master_feature_addlhar.csv')
        df.to_csv(df_path, index=False)
        
        return df

    else:
        print(f"❌ Could not find {master_path}. Run master_pipeline.py first.")

    
if __name__ == "__main__":
    add_lhar_main()
    