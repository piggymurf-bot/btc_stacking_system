import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import os
import re

from src.config_v2 import FEATURE_GROUPS

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]

def run_alpha_walk_forward_optimization(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    target_col: str = "target_fwd_cum_ret_14d",
    target_col_bin: str = "target_fwd_dir_14d", 
    train_window_years: int = 1,
    test_window_months: int = 3
):
    print(f"📥 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
        
    feature_cols = [
        col for col in df.columns 
        if any(col.startswith(prefix) for prefix in ONCHAIN_FEATURES)
        and not col.startswith("target_")
        and col not in ["date", "close", "open", "high", "low", "volume", "vwap"]
    ]
    
    # --- FIX: Clean feature names to remove special JSON characters for LightGBM ---
    cleaned_feature_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in feature_cols]
    
    # Assign cleaned names to the dataframe slices for this run
    df_clean = df.dropna(subset=[target_col, target_col_bin] + cleaned_feature_cols).reset_index(drop=True)
    df_clean = df_clean.rename(columns=dict(zip(feature_cols, cleaned_feature_cols)))
    feature_cols = cleaned_feature_cols
    # -----------------------------------------------------------------------------
    
    # Define date bounds for rolling splits
    min_date = df_clean["date"].min()
    max_date = df_clean["date"].max()
    
    print(f"⚙️ Starting WFO from {min_date.date()} to {max_date.date()}...")
    
    # Approximate days per window
    train_days = train_window_years * 365
    test_days = test_window_months * 30
    
    all_oos_preds = []
    
    current_start_idx = 0
    total_rows = len(df_clean)
    fold_idx = 1
    
    while True:
        train_end_idx = current_start_idx + train_days
        test_end_idx = train_end_idx + test_days
        
        if train_end_idx >= total_rows:
            break
            
        if test_end_idx > total_rows:
            test_end_idx = total_rows
            
        train_slice = df_clean.iloc[current_start_idx:train_end_idx]
        test_slice = df_clean.iloc[train_end_idx:test_end_idx]
        
        if len(test_slice) < 30: # Skip tiny trailing fragments
            break
            
        X_train, y_train, y_train_b = train_slice[feature_cols], train_slice[target_col], train_slice[target_col_bin]
        X_test, y_test, y_test_b = test_slice[feature_cols], test_slice[target_col], test_slice[target_col_bin]
        
        # Inside your WFO loop, right after getting test_slice:
        print(f"\n--- DIAGNOSTIC FOR FOLD {fold_idx} ({test_slice['date'].min().date()} to {test_slice['date'].max().date()}) ---")
        
        # Check target distribution in test set
        print(f"Target Mean: {y_test.mean():.4f} | Target Std: {y_test.std():.4f}")
        
        # Check top feature correlation with target *only* on this test fold
        fold_ic_results = {}
        for col in feature_cols:
            corr, _ = spearmanr(test_slice[col], y_test)
            fold_ic_results[col] = corr
            
        fold_ic_df = pd.DataFrame(list(fold_ic_results.items()), columns=["Feature", "Test_IC"]).sort_values(by="Test_IC", key=abs, ascending=False)
        print("Top 5 Features correlated with Target in this Test Fold:")
        print(fold_ic_df.head(5))
        
        # LightGBM Parameters (locked in from your validated RMSE setup)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "min_child_samples": 30,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1
        }
        
        # Train Base Classifier (Manitude Alpha)
        reg_model = lgb.LGBMRegressor(**params)
        reg_model.fit(
            X_train, y_train,
            eval_X = X_test,
            eval_y = y_test,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds_reg = reg_model.predict(X_test)
        
        
        # Train Base Classifier (Direction Probability)
        clf_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
        clf_model.fit(X_train, y_train_b)
        preds_prob = clf_model.predict_proba(X_test)[:, 1] # Probability of class 1 (Up)
            
        # --- ROBUST EVALUATION BLOCK ---
        clean_preds = np.asarray(preds_reg, dtype=float)
        clean_y = np.asarray(y_test, dtype=float)
        
        # Drop any NaN pairs
        valid_mask = ~(np.isnan(clean_preds) | np.isnan(clean_y))
        clean_preds = clean_preds[valid_mask]
        clean_y = clean_y[valid_mask]
        
        # Check for empty arrays, zero variance, or constant predictions
        if len(clean_preds) < 2 or np.all(clean_preds == clean_preds[0]) or np.all(clean_y == clean_y[0]):
            ic = 0.0
        else:
            res = spearmanr(clean_preds, clean_y)
            ic = res.correlation if not np.isnan(res.correlation) else 0.0
        # -------------------------------
        
        # Store results for this fold using the safe 'ic' value
        fold_results = pd.DataFrame({
            "date": test_slice["date"].iloc[valid_mask] if len(valid_mask) == len(test_slice) else test_slice["date"],
            "y_true": clean_y,
            "y_pred": clean_preds,
            "fold": fold_idx
        })
        all_oos_preds.append(fold_results)
        
        print(f"--- WFO Fold {fold_idx} ({test_slice['date'].min().date()} to {test_slice['date'].max().date()}) ---")
        print(f"📊 Out-of-Sample Spearman IC: {ic:.4f}")
        
        # Roll forward by test window size
        current_start_idx += test_days
        fold_idx += 1
        
    # Stitch all out-of-sample predictions together
    master_oos_df = pd.concat(all_oos_preds, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/alpha_wfo_out_of_sample_predictions.csv"
    master_oos_df.to_csv(output_file, index=False)
    
    # Calculate overall stitched metrics
    overall_ic, _ = spearmanr(master_oos_df["y_pred"], master_oos_df["y_true"])
    overall_dir_acc = (np.sign(master_oos_df["y_pred"]) == np.sign(master_oos_df["y_true"])).mean()
    
    print("\n========================================")
    print("🔒 STITCHED WALK-FORWARD PERFORMANCE")
    print(f"📊 Overall Stitched Spearman IC : {overall_ic:.4f}")
    print(f"📊 Overall Directional Accuracy : {overall_dir_acc * 100:.2f}%")
    print(f"📁 Saved to: {output_file}")
    print("========================================")
    
    # --- LEAK DIAGNOSTIC SNIPPET ---
    # 1. Check Feature Importances: Does a single feature have 99% importance?
    importances = reg_model.feature_importances_
    top_feature_idx = importances.argsort()[-1]
    print(f"🚨 Top Feature driving 0.99 IC: {feature_cols[top_feature_idx]} (Importance: {importances[top_feature_idx]})")

    # 2. Check if y_pred correlates 1:1 with a specific feature instead of the model's complex tree splits
    # If the top feature's correlation with y_true is also 0.99, that feature is leaking the target!
    feature_corr, _ = spearmanr(X_test.iloc[:, top_feature_idx], y_test)
    print(f"🚨 Top Feature correlation with target: {feature_corr:.4f}")
    #---------------------------------

if __name__ == "__main__":
    run_alpha_walk_forward_optimization()