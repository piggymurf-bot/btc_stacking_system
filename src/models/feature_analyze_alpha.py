import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.config_v2 import FEATURE_GROUPS
from src.features.fracdiff import frac_diff_ffd, find_optimal_d

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]

def analyze_fold_feature_importance_alpha(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    target_col: str = "target_fwd_sharpe_7d"
):
    """
    Trains LightGBM across all 5 TimeSeriesSplit folds and tracks 
    feature importance shifts (Gain and Split) from fold to fold.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    print(f"📥 Loading dataset for feature importance audit: {data_path}")
    df = pd.read_csv(data_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    
    non_stationary_cols = ['aviv', 'mvrv', 'nupl', 'nrplbtc_log', 'close']
    d_opt_df = pd.DataFrame(columns=non_stationary_cols, index=range(1))
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,                 # Keep depth at 4 so small folds can form valid splits
        "num_leaves": 15,               
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,        # Balanced feature fraction (lets features cooperate)
        "min_child_samples": 30,        # CRITICAL: Ensures every leaf has at least 30 samples, preventing micro-overfitting on early folds
        "reg_alpha": 0.2,               # Light regularization
        "reg_lambda": 1.0,              
        "random_state": 42,
        "verbose": -1
    }   

    importance_records = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        print(f"\n🔄 Processing Fold {fold + 1} (Train rows: {len(train_df)}, Val rows: {len(val_df)})")

        # Apply fractional differentiation on non-stationary columns using train bounds
        for col in non_stationary_cols:
            series_train = np.log(train_df[col]) if col == 'close' or (train_df[col] > 0).all() else train_df[col]
            series_val = np.log(val_df[col]) if col == 'close' or (val_df[col] > 0).all() else val_df[col]

            if fold == 0:
                optimal_d = find_optimal_d(series_train, p_thres=0.05)
                d_opt_df[f'{col}'] = optimal_d
            else:
                optimal_d = d_opt_df.loc[0, f'{col}']
            
            train_df[f'{col}_fracdiff'] = frac_diff_ffd(series_train, d=optimal_d)
            val_df[f'{col}_fracdiff'] = frac_diff_ffd(series_val, d=optimal_d)

        available_features = [f for f in train_df.columns if f in ONCHAIN_FEATURES or f.endswith('_fracdiff')]
        
        # Drop the over-dominant feature to force multi-factor learning
        if 'mvrv_daily_diff' in available_features:
            available_features.remove('mvrv_daily_diff')
        
        train_df = train_df.dropna(subset=[target_col] + available_features).reset_index(drop=True)
        X_train, y_train = train_df[available_features], train_df[target_col]

        # Train model for this fold
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # Extract importances (Gain and Split)
        gain_importance = model.booster_.feature_importance(importance_type='gain')
        split_importance = model.booster_.feature_importance(importance_type='split')

        for feat, gain, split in zip(available_features, gain_importance, split_importance):
            importance_records.append({
                "fold": fold + 1,
                "feature": feat,
                "gain": gain,
                "split": split
            })

    # Compile into a comparative DataFrame
    imp_df = pd.DataFrame(importance_records)
    
    # Pivot to view features as rows and folds as columns (using 'gain')
    pivot_gain = imp_df.pivot(index="feature", columns="fold", values="gain").fillna(0)
    # Normalize each fold to percentage importance (sum to 100)
    pivot_gain_pct = pivot_gain.div(pivot_gain.sum(axis=0), axis=1) * 100
    
    # Sort by average importance across folds
    pivot_gain_pct["mean_importance"] = pivot_gain_pct.mean(axis=1)
    pivot_gain_pct = pivot_gain_pct.sort_values(by="mean_importance", ascending=False)

    print("\n" + "="*70)
    print("📊 TOP 10 FEATURE IMPORTANCE SHIFTS ACROSS FOLDS (% GAIN)")
    print("="*70)
    print(pivot_gain_pct.drop(columns=["mean_importance"]).head(10).round(2))
    print("="*70)

    # Save to CSV for full review
    out_path = "data/processed/fold_feature_importance_audit_alpha.csv"
    pivot_gain_pct.to_csv(out_path)
    print(f"📁 Full feature importance audit saved to: {out_path}")

if __name__ == "__main__":
    analyze_fold_feature_importance_alpha()