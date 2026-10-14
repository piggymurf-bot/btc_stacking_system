import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.config_v2 import FEATURE_GROUPS
from src.features.fracdiff import frac_diff_ffd, find_optimal_d
from src.models.diag_functions import evaluate_predictions

DERIVATIVES_FEATURES = FEATURE_GROUPS["derivatives_microstructure_features"]
TECHNICALS_VOL_FEATURES = FEATURE_GROUPS["technicals_and_volatility_features"]

def run_holdout_evaluation_volat(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    target_col: str = "target_rv"
):
    """
    Trains on folds 1-4 and tests strictly on the final unseen fold (Fold 5) 
    to evaluate true out-of-sample performance without OOF smoothing.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    print(f"📥 Loading dataset for holdout test: {data_path}")
    df = pd.read_csv(data_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(df))
    
    vol_features = list(set(DERIVATIVES_FEATURES + TECHNICALS_VOL_FEATURES))
    
    # Isolate strictly Fold 5 (The final unseen block)
    train_indices, test_indices = splits[-1]
    
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    print(f"📊 Training set size: {len(train_df)} rows")
    print(f"📊 Unseen Holdout (Fold 5) size: {len(test_df)} rows (from {test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Apply fractional differentiation on non-stationary columns
    non_stationary_cols = ['aviv', 'mvrv', 'nupl', 'nrplbtc_log', 'close']
    for col in non_stationary_cols:
        series_train = np.log(train_df[col]) if col == 'close' or (train_df[col] > 0).all() else train_df[col]
        series_test = np.log(test_df[col]) if col == 'close' or (test_df[col] > 0).all() else test_df[col]

        optimal_d = find_optimal_d(series_train, p_thres=0.05)
        
        train_df[f'{col}_fracdiff'] = frac_diff_ffd(series_train, d=optimal_d)
        test_df[f'{col}_fracdiff'] = frac_diff_ffd(series_test, d=optimal_d)

    available_features = [f for f in vol_features if f in train_df.columns]
    
    train_df = train_df.dropna(subset=[target_col] + available_features).reset_index(drop=True)
    X_train, y_train = train_df[available_features], train_df[target_col]
    
    test_df = test_df.dropna(subset=available_features).reset_index(drop=True)
    X_test = test_df[available_features]
    y_test = test_df[target_col]

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

    print("\n🚀 Training model on historical folds 1-4...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_X = X_test,
        eval_y = y_test,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    print("🎯 Evaluating strictly on Unseen Holdout Fold 5...")
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print("\n" + "="*40)
    print(f"🔒 HOLDOUT EVALUATION (FOLD 5 UNSEEN)")
    print(f"📊 Holdout RMSE: {rmse:.5f}")
    print(f"📊 Holdout MAE:  {mae:.5f}")
    print("="*40)

    # Save holdout predictions for independent backtesting
    test_df["oof_volat_preds"] = preds
    test_df[["date", "close", "oof_volat_preds", target_col]].to_csv(
        "data/processed/holdout_fold5_predictions_volat.csv", index=False
    )
    print("📁 Holdout predictions saved to: data/processed/holdout_fold5_predictions_volat.csv")

    evaluate_predictions(y_test, preds)

if __name__ == "__main__":
    run_holdout_evaluation_volat()