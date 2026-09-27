import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.config_v2 import FEATURE_GROUPS

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]

def train_alpha_model(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    target_col: str = "target_fwd_cum_ret_1d"
):
    """
    Trains a LightGBM Regressor on Group 1 On-Chain Features using Walk-Forward TimeSeriesSplit.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run feature pipelines first.")

    print(f"📥 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # 1. Isolate Features (X) and Target (y)
    # Ensure all selected features actually exist in the dataframe
    available_features = [f for f in ONCHAIN_FEATURES if f in df.columns]
    missing_check = set(ONCHAIN_FEATURES) - set(available_features)
    if missing_check:
        print(f"⚠️ Warning: The following features were missing and dropped: {missing_check}")

    # Drop rows with NaNs in features or target
    model_df = df[available_features + [target_col, "date"]].dropna().reset_index(drop=True)
    
    X = model_df[available_features]
    y = model_df[target_col]
    dates = model_df["date"]

    print(f"🎯 Training target: {target_col}")
    print(f"📊 Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")

    # 2. Walk-Forward Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_preds = np.zeros(len(model_df))
    fold_scores = []

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize and fit LightGBM
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            #eval_set=[(X_val, y_val)],
            eval_X = X_val,
            eval_y = y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Predict on validation fold
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append(fold_rmse)
        print(f"  -> Fold {fold+1} | Train Size: {len(X_train)} | Val Size: {len(X_val)} | RMSE: {fold_rmse:.5f}")

    # Overall OOF Evaluation (ignoring the initial warm-up split indices that remain 0)
    valid_mask = oof_preds != 0
    overall_rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_preds[valid_mask]))
    overall_mae = mean_absolute_error(y[valid_mask], oof_preds[valid_mask])
    
    print("\n" + "="*40)
    print("✅ Group 1 LightGBM Training Complete!")
    print(f"📊 Overall Out-of-Fold RMSE: {overall_rmse:.5f}")
    print(f"📊 Overall Out-of-Fold MAE:  {overall_mae:.5f}")
    print("="*40)

    # 3. Feature Importance Analysis
    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 5 Most Predictive On-Chain Features for 1-Day Return:")
    print(importance_df.head(5).to_string(index=False))

    return model, importance_df

if __name__ == "__main__":
    train_alpha_model()