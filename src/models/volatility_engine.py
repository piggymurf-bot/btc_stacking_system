import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.config_v2 import FEATURE_GROUPS
from src.features.fracdiff import frac_diff_ffd,find_optimal_d

DERIVATIVES_FEATURES = FEATURE_GROUPS["derivatives_microstructure_features"]
TECHNICALS_VOL_FEATURES = FEATURE_GROUPS["technicals_and_volatility_features"]

def train_volatility_model(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    output_model_path: str = "models/",
    output_path: str = "data/processed",
    target_col: str = "target_rv"
):
    """
    Trains a LightGBM Volatility Prediction Engine using Derivatives and Technical/LHAR features.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}.")

    print(f"📥 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Combine Group 2 and Group 3 features for total volatility coverage
    vol_features = list(set(DERIVATIVES_FEATURES + TECHNICALS_VOL_FEATURES))
    
    y = df[target_col]
   
    # Walk-Forward Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
   
    oof_preds = np.zeros(len(df))
    fold_scores = []

    params = {
       "objective": "regression",
       "metric": "rmse",
       "boosting_type": "gbdt",
       "n_estimators": 500,
       "learning_rate": 0.03,
       "max_depth": 4,
       "num_leaves": 15,
       "subsample": 0.7,
       "subsample_freq": 1,        # Enables row resampling per iteration
       "colsample_bytree": 0.5,    # Encourages broader feature usage
       "reg_alpha": 0.1,           # L1 regularization for sparse noise reduction
       "reg_lambda": 1.0,          # L2 regularization to control weight magnitudes
       "random_state": 42,
       "verbose": -1
    }

    # List of cumulative non-stationary features to transform
    non_stationary_cols = [ 'aviv', 'mvrv', 'nupl', 'nrplbtc_log', 'close']
   
    # Assign the DataFrame to collect optimum d
    d_opt_df = pd.DataFrame(columns=non_stationary_cols,index=range(1))
   
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        for col in non_stationary_cols:

            # Prepare the series (applying log if close or positive values)
            series_train = np.log(train_df[col]) if col == 'close' or (train_df[col] > 0).all() else train_df[col]
            series_val = np.log(val_df[col]) if col == 'close' or (val_df[col] > 0).all() else val_df[col]

            # CONDITION: Optimize on Fold 0 (the first fold), lock and reuse for later folds
            if fold == 0:
                optimal_d = find_optimal_d(series_train, p_thres=0.05)
                d_opt_df[f'{col}'] = optimal_d  # Collect the optimal d for each column
                #print(f"[Fold {fold}] Calculated and locked d* for {col}: {optimal_d}")
            else:
                optimal_d = d_opt_df.loc[0,f'{col}']  # Reuse the locked value
                #print(f"[Fold {fold}] Reusing locked d* for {col}: {optimal_d}")
            
            # Apply fractional differentiation using the determined d
            train_df[f'{col}_fracdiff'] = frac_diff_ffd(series_train, d=optimal_d)
            val_df[f'{col}_fracdiff'] = frac_diff_ffd(series_val, d=optimal_d)

        # Ensure all selected features actually exist in the dataframe
        available_features = [f for f in vol_features if f in train_df.columns]
        missing_check = set(vol_features) - set(available_features)
        if missing_check:
            print(f"⚠️ Warning: The following features were missing and dropped: {missing_check}")

        X_train, X_val = train_df[available_features], val_df[available_features]
        y_train, y_val = train_df[target_col], val_df[target_col]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            #eval_set=[(X_val, y_val)],
            eval_X = X_val,
            eval_y = y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

         # Save individual fold artifacts (just the model and feature list)
        fold_artifact = {
             "model": model,
             "features": available_features
        }
        artifact_path = os.path.join(output_model_path, f"fold_{fold+1}_volat.pkl")
        joblib.dump(fold_artifact, artifact_path)

        # Predict on validation fold
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append(fold_rmse)
        print(f"  -> Fold {fold+1} | Train Size: {len(X_train)} | Val Size: {len(X_val)} | RMSE: {fold_rmse:.5f}")

    # Evaluate overall out-of-fold performance
    valid_mask = oof_preds != 0
    overall_rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_preds[valid_mask]))
    overall_mae = mean_absolute_error(y[valid_mask], oof_preds[valid_mask])

    print("\n" + "="*45)
    print(f"✅ Volatility Engine Training Complete!")
    print(f"📊 Overall Out-of-Fold RMSE: {overall_rmse:.5f}")
    print(f"📊 Overall Out-of-Fold MAE:  {overall_mae:.5f}")
    print("="*45)

    # Display Top 10 Most Predictive Volatility Features
    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Most Predictive Volatility Features:")
    print(importance_df.head(10).to_string(index=False))
        
    # Save OOF for backtest
    # Save OOF for backtest
    oof_df = pd.DataFrame(
        data=oof_preds, 
        columns=['oof_volat_preds']
    )
    oof_path = os.path.join(output_path, 'oof_volat.csv')
    oof_df.to_csv(oof_path, index=True)
    
    # Save optimal d for processing daily dataset 
    d_opt_path = os.path.join('data/processed', 'optimal_d_volat.csv')
    d_opt_df.to_csv(d_opt_path, index=True)

    return model, importance_df

if __name__ == "__main__":
    train_volatility_model()