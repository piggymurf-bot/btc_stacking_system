import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
import os
import re

from src.config_v2 import FEATURE_GROUPS

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]

def run_alpha_stack_walk_forward_optimization(
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
    
    # Define a buffer offset (e.g., 1.5% expected move over 14 days)
    OFFSET = 0.015

    # Create binary target, but ignore the noisy middle zone
    # Class 1: Strong Up (> +1.5%)
    # Class 0: Strong Down (< -1.5%)
    df_clean = df_clean[
        (df_clean[target_col] > OFFSET) | (df_clean[target_col] < -OFFSET)
    ].copy()

    df_clean["target_fwd_dir_14d_offset"] = (df_clean[target_col] > 0).astype(int)
    
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
            
        X_train, y_train, y_train_b = train_slice[feature_cols], train_slice[target_col], train_slice["target_fwd_dir_14d_offset"]
        X_test, y_test, y_test_b = test_slice[feature_cols], test_slice[target_col], test_slice["target_fwd_dir_14d_offset"]
        
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
        
        
        clf_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,             # Keep shallow to prevent noise fitting
            "num_leaves": 15,           # Consistent with your regressor
            "subsample": 0.8,           # Row subsampling for generalization
            "subsample_freq": 1,
            "colsample_bytree": 0.7,    # Feature subsampling per tree
            "min_child_samples": 30,    # Ensures enough samples per leaf node
            "reg_alpha": 0.2,           # L1 regularization
            "reg_lambda": 1.0,          # L2 regularization
            "random_state": 42,
            "verbose": -1
        }
        
        # Train Base Classifier (Direction Probability)
        clf_model = lgb.LGBMClassifier(**clf_params)
        clf_model.fit(
            X_train, y_train_b,
            eval_X = X_test,
            eval_y = y_test_b,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        preds_prob = clf_model.predict_proba(X_test)[:, 1] # Probability of class 1 (Up)
            
        # Store OOF predictions for this fold
        fold_results = pd.DataFrame({
            "date": test_slice["date"],
            "y_true_reg": y_test,
            "y_true_bin": y_test_b.values,
            "pred_reg": preds_reg,
            "pred_prob": preds_prob,
            "fold": fold_idx
        })
        all_oos_preds.append(fold_results)
                
        # Roll forward by test window size
        current_start_idx += test_days
        fold_idx += 1
    
    master_oos_df = pd.concat(all_oos_preds, ignore_index=True)
    master_oos_df = master_oos_df.sort_values("date").reset_index(drop=True)

    target_meta_y = "y_true_reg" 

    unique_folds = sorted(master_oos_df["fold"].unique())
    meta_oos_results = []

    print("\n🚀 Starting Walk-Forward Evaluation for Ridge Meta-Learner...")

    # Need at least 2 folds of history before we can train and test sequentially
    for i in range(2, len(unique_folds)):
        train_folds = unique_folds[:i]
        test_fold = unique_folds[i]
        
        train_meta_df = master_oos_df[master_oos_df["fold"].isin(train_folds)]
        test_meta_df = master_oos_df[master_oos_df["fold"] == test_fold]

        if len(test_meta_df) == 0:
            continue

        # Meta-Features: Base Regressor prediction & Base Classifier probability
        X_meta_train = train_meta_df[["pred_reg", "pred_prob"]]
        y_meta_train = train_meta_df[target_meta_y]

        X_meta_test = test_meta_df[["pred_reg", "pred_prob"]]
        y_meta_test = test_meta_df[target_meta_y]

        # 1. Train Continuous Meta-Regressor (Ridge)
        meta_model = Ridge(alpha=0.01)
        meta_model.fit(X_meta_train, y_meta_train)
        
        # meta_params = {
        #     "objective": "regression",
        #     "metric": "mae",  
        #     "boosting_type": "gbdt",
        #     "n_estimators": 50,
        #     "learning_rate": 0.05,
        #     "max_depth": 2,
        #     "num_leaves": 3,
        #     "subsample": 0.8,
        #     "colsample_bytree": 1.0,
        #     "min_child_samples": 20,
        #     "random_state": 42,
        #     "verbose": -1
        # }

        # meta_model = lgb.LGBMRegressor(**meta_params)
        # meta_model.fit(X_meta_train, y_meta_train)
        
        # 2. Predict out-of-sample probability and derive binary class
        meta_preds = meta_model.predict(X_meta_test)
        
        # Store results for this test fold
        fold_eval = test_meta_df[["date", "y_true_reg", "y_true_bin", "pred_reg", "pred_prob"]].copy()
        fold_eval["meta_pred_reg"] = meta_preds
        # Derive binary direction from the continuous meta-prediction (> 0 is Up)
        fold_eval["meta_pred_bin"] = (meta_preds > 0).astype(int)
        fold_eval["meta_fold_tested"] = test_fold

        meta_oos_results.append(fold_eval)

    # Combine all truly OOS meta-predictions
    if meta_oos_results:
        master_meta_oos = pd.concat(meta_oos_results, ignore_index=True)
        
        # Evaluate Directional Accuracy of the Meta-Learner
        meta_dir_acc = accuracy_score(master_meta_oos["y_true_bin"], master_meta_oos["meta_pred_bin"])
        base_reg_dir_acc = accuracy_score(master_meta_oos["y_true_bin"], (master_meta_oos["pred_reg"] > 0).astype(int))
        base_clf_dir_acc = accuracy_score(master_meta_oos["y_true_bin"], (master_meta_oos["pred_prob"] > 0.5).astype(int))

        print("\n========================================")
        print("🎯 TRUE OOS RIDGE META-LEARNER PERFORMANCE")
        print(f"📊 Base Regressor Direction Accuracy  : {base_reg_dir_acc * 100:.2f}%")
        print(f"📊 Base Classifier Direction Accuracy : {base_clf_dir_acc * 100:.2f}%")
        print(f"🚀 Staked Ridge Meta Direction Accuracy : {meta_dir_acc * 100:.2f}%")
        print("========================================")

        # Save final OOS stacked dataset
        os.makedirs("data/processed", exist_ok=True)
        output_file = "data/processed/alpha_stack_wfo_out_of_sample_predictions.csv"
        master_meta_oos.to_csv(output_file, index=False)
        print(f"📁 Ridge meta-predictions saved to: {output_file}")
    else:
        print("⚠️ Not enough folds to run sequential meta-evaluation.")
    
    
    # # Calculate overall stitched metrics
    # overall_ic, _ = spearmanr(master_oos_df["y_pred"], master_oos_df["y_true"])
    # overall_dir_acc = (np.sign(master_oos_df["y_pred"]) == np.sign(master_oos_df["y_true"])).mean()
    
    # print("\n========================================")
    # print("🔒 STITCHED WALK-FORWARD PERFORMANCE")
    # print(f"📊 Overall Stitched Spearman IC : {overall_ic:.4f}")
    # print(f"📊 Overall Directional Accuracy : {overall_dir_acc * 100:.2f}%")
    # print(f"📁 Saved to: {output_file}")
    # print("========================================")
    
    # # --- LEAK DIAGNOSTIC SNIPPET ---
    # # 1. Check Feature Importances: Does a single feature have 99% importance?
    # importances = reg_model.feature_importances_
    # top_feature_idx = importances.argsort()[-1]
    # print(f"🚨 Top Feature driving 0.99 IC: {feature_cols[top_feature_idx]} (Importance: {importances[top_feature_idx]})")

    # # 2. Check if y_pred correlates 1:1 with a specific feature instead of the model's complex tree splits
    # # If the top feature's correlation with y_true is also 0.99, that feature is leaking the target!
    # feature_corr, _ = spearmanr(X_test.iloc[:, top_feature_idx], y_test)
    # print(f"🚨 Top Feature correlation with target: {feature_corr:.4f}")
    # #---------------------------------

if __name__ == "__main__":
    run_alpha_stack_walk_forward_optimization()