import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.config_v2 import FEATURE_GROUPS
from src.features.fracdiff import frac_diff_ffd, find_optimal_d

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]

def train_alpha_model_ridge(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    target_col: str = "target_fwd_cum_ret_1d"
):
    """
    Trains a Ridge Regression Baseline on Group 1 On-Chain Features using Walk-Forward TimeSeriesSplit
    with safe boolean-masked validation tracking.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run feature pipelines first.")

    print(f"📥 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    y = df[target_col]
    
    # Walk-Forward Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_preds = np.zeros(len(df))
    fold_scores = []
    fold_coefficients = []

    # List of cumulative non-stationary features to transform
    non_stationary_cols = ['aviv', 'mvrv', 'nupl', 'nrplbtc_log', 'close']
    
    # Assign the DataFrame to collect optimum d
    d_opt_df = pd.DataFrame(columns=non_stationary_cols, index=range(1))
    
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
            else:
                optimal_d = d_opt_df.loc[0, f'{col}']  # Reuse the locked value
            
            # Apply fractional differentiation using the determined d
            train_df[f'{col}_fracdiff'] = frac_diff_ffd(series_train, d=optimal_d)
            val_df[f'{col}_fracdiff'] = frac_diff_ffd(series_val, d=optimal_d)

        # Ensure all selected features actually exist in the dataframe
        available_features = [f for f in ONCHAIN_FEATURES if f in train_df.columns]
        missing_check = set(ONCHAIN_FEATURES) - set(available_features)
        if missing_check:
            print(f"⚠️ Warning: The following features were missing and dropped: {missing_check}")

        # Drop NaNs from training set normally
        train_df = train_df.dropna(subset=[target_col] + available_features).reset_index(drop=True)
        
        X_train = train_df[available_features]
        y_train = train_df[target_col]

        # 1. Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 2. Initialize and fit Ridge Regression
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 3. Handle validation NaNs and alignment via boolean masking
        val_eval_cols = available_features + [target_col]
        val_mask = val_df[val_eval_cols].notna().all(axis=1)
        
        if val_mask.sum() > 0:
            val_clean = val_df[val_mask].copy()
            X_val = val_clean[available_features]
            y_val = val_clean[target_col]
            
            X_val_scaled = scaler.transform(X_val)
            preds = model.predict(X_val_scaled)

            # Map predictions back to the exact global index positions safely
            global_val_indices = val_df.index[val_mask]
            oof_preds[global_val_indices] = preds

            fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_scores.append(fold_rmse)
            fold_coefficients.append(model.coef_)
            print(f"  -> Fold {fold+1} | Train Size: {len(X_train)} | Val Size (Clean): {len(X_val)} | RMSE: {fold_rmse:.5f}")
        else:
            print(f"  -> Fold {fold+1} | Train Size: {len(X_train)} | Val Size (Clean): 0 | Warning: All validation rows are NaN")

    # Overall OOF Evaluation (ignoring unpredicted indices)
    valid_mask = oof_preds != 0
    overall_rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_preds[valid_mask]))
    overall_mae = mean_absolute_error(y[valid_mask], oof_preds[valid_mask])
    
    print("\n" + "="*40)
    print(f"✅ Group 1 Ridge Baseline Training Complete!")
    print(f"📊 Overall Out-of-Fold RMSE: {overall_rmse:.5f}")
    print(f"📊 Overall Out-of-Fold MAE:  {overall_mae:.5f}")
    print("="*40)

    # Average coefficients across folds to inspect linear feature impact
    if fold_coefficients:
        mean_coefs = np.mean(fold_coefficients, axis=0)
        coef_df = pd.DataFrame({
            "feature": available_features,
            "coefficient": mean_coefs,
            "abs_coefficient": np.abs(mean_coefs)
        }).sort_values(by="abs_coefficient", ascending=False)

        print("\nTop 5 Linear Drivers (by Absolute Mean Ridge Coefficient):")
        print(coef_df.drop(columns=["abs_coefficient"]).head(5).to_string(index=False))
    else:
        coef_df = pd.DataFrame()

    # Save optimal d
    d_opt_path = os.path.join('data/processed', 'optimal_d_v2.csv')
    d_opt_df.to_csv(d_opt_path, index=True)
    
    return model, coef_df

if __name__ == "__main__":
    train_alpha_model_ridge()