import json
import os
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.models.v2_features_partition import determine_market_regime

def run_regime_stacked_serialized_pipeline(
    master_df: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str, 
    lgb_params: dict,
    meta_alpha: float,
    val_block_size: int = 246,
    holdout_size: int = 150,
    profile_dir: str = "data/regime_profiles",
    artifact_dir: str = "models/regime_stacked_artifacts",
    experiment_name: str = "Bitcoin_Regime_Stacked_Serialized",
):
    """
    Trains regime experts and meta-learner in WFO folds, serializes them, 
    and evaluates them strictly on the untouched holdout block.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"Serialized_Stacking_{target_col}"):
        print(f"🚀 Starting Serialized Regime-Stacked Pipeline for {target_col}...")
        
        holdout_df = master_df.iloc[-holdout_size:].copy()
        wfo_df = master_df.iloc[:-holdout_size].copy()
        
        print(f"📊 Total Dataset Rows: {len(master_df)}")
        print(f"   - WFO Training/Validation Pool: {len(wfo_df)} rows")
        print(f"   - Isolated Holdout Set:         {len(holdout_df)} rows")

        regimes = []
        window = 14
        for i in range(len(wfo_df)):
            slice_df = wfo_df.iloc[max(0, i - window + 1):i + 1]
            regimes.append(determine_market_regime(slice_df, window=window))
        
        wfo_df = wfo_df.copy()
        wfo_df['market_regime'] = regimes
        
        # Optuna LightGBM parameters for base experts
        tuned_lgb_params = lgb_params

        # Optuna Ridge parameter for the meta-learner
        tuned_meta_alpha = meta_alpha
        
        # Print exact distribution to verify partitioning
        print("\n📊 Historical Regime Distribution:")
        print(wfo_df['market_regime'].value_counts())
        print("-" * 50)
        
        oof_meta_preds = []
        oof_actuals = []
        total_rows = len(wfo_df)
        regimes = ['clean_trend', 'high_vol_chop', 'mean_reverting_sideways']
        
        # 1. Run WFO Folds
        for fold_idx in range(1, 4):
            print("\n------------------------------------------------------------")
            print(f"   -> Processing WFO Fold {fold_idx}...")
            
            horizon = int(target_col.split('_')[-1].replace('d', ''))
            train_end_idx = total_rows - (4 - fold_idx) * val_block_size - horizon
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + val_block_size, total_rows)
            
            train_df = wfo_df.iloc[:train_end_idx].copy()
            val_df = wfo_df.iloc[val_start_idx:val_end_idx].copy()
            
            if len(val_df) < 30:
                break
                
            y_train = train_df[target_col]
            y_val = val_df[target_col]
            
            # Train Base Regime Experts
            expert_models = {}
            for regime in regimes:
                profile_path = os.path.join(profile_dir, f"regime_profile_{regime}.json")
                if not os.path.exists(profile_path):
                    continue
                with open(profile_path, "r") as f:
                    profile = json.load(f)
                    
                features = profile["core_features"] + profile["satellite_pool"]
                regime_train_df = train_df[train_df['market_regime'] == regime]
                if len(regime_train_df) < 30:
                    regime_train_df = train_df
                    
                X_tr_regime = regime_train_df[[f for f in features if f in train_df.columns]]
                y_tr_regime = train_df.loc[X_tr_regime.index, target_col]
                
                lgb_train = lgb.Dataset(X_tr_regime, label=y_tr_regime)
                # model = lgb.train({'objective': 'regression', 'learning_rate': 0.02, 'verbose': -1}, lgb_train, num_boost_round=120)
                model = lgb.train(tuned_lgb_params, lgb_train, num_boost_round=100)
                expert_models[regime] = {"model": model, "features": X_tr_regime.columns.tolist()}
            
            def get_expert_matrix(df):
                matrix = np.zeros((len(df), len(regimes)))
                for idx, regime in enumerate(regimes):
                    if regime in expert_models:
                        exp = expert_models[regime]
                        X_sub = df[[f for f in exp["features"] if f in df.columns]]
                        matrix[:, idx] = exp["model"].predict(X_sub)
                return matrix
                
            train_meta_X = get_expert_matrix(train_df)
            val_meta_X = get_expert_matrix(val_df)
            
            # Train Meta-Learner
            meta_learner = Ridge(alpha = tuned_meta_alpha)
            meta_learner.fit(train_meta_X, y_train)

            print(f"Meta-Learner Expert Weights: {dict(zip(regimes, meta_learner.coef_))}")
            
            meta_preds = meta_learner.predict(val_meta_X)
            fold_ic, _ = spearmanr(meta_preds, y_val)
            print(f"   -> Fold {fold_idx} Stacked IC: {fold_ic:.4f}")
            
            oof_meta_preds.extend(meta_preds)
            oof_actuals.extend(y_val.values)
            
            # If this is the final fold (Fold 3), serialize these exact artifacts for holdout testing
            if fold_idx == 3:
                print(f"💾 Serializing Fold 3 artifacts to {artifact_dir}...")
                joblib.dump(expert_models, os.path.join(artifact_dir, "fold3_expert_models.pkl"))
                joblib.dump(meta_learner, os.path.join(artifact_dir, "fold3_meta_learner.pkl"))
                
        aggregate_ic, _ = spearmanr(oof_meta_preds, oof_actuals)
        print(f"\n🎯 WFO Aggregate OOF Spearman IC: {aggregate_ic:.4f}")
        
        # 2. Evaluate Untouched Holdout Using Serialized Fold 3 Artifacts
        print("\n" + "="*50)
        print("🏆 EVALUATING UNTOUCHED HOLDOUT WITH SERIALIZED FOLD 3 ARTIFACTS")
        print("="*50)
        
        loaded_experts = joblib.load(os.path.join(artifact_dir, "fold3_expert_models.pkl"))
        loaded_meta = joblib.load(os.path.join(artifact_dir, "fold3_meta_learner.pkl"))
        
        def get_loaded_expert_matrix(df):
            matrix = np.zeros((len(df), len(regimes)))
            for idx, regime in enumerate(regimes):
                if regime in loaded_experts:
                    exp = loaded_experts[regime]
                    X_sub = df[[f for f in exp["features"] if f in df.columns]]
                    matrix[:, idx] = exp["model"].predict(X_sub)
            return matrix
            
        holdout_meta_X = get_loaded_expert_matrix(holdout_df)
        holdout_preds = loaded_meta.predict(holdout_meta_X)
        
        holdout_ic, holdout_pval = spearmanr(holdout_preds, holdout_df[target_col])
        mlflow.log_metric("serialized_holdout_spearman_ic", holdout_ic)
        mlflow.log_metric("serialized_holdout_p_value", holdout_pval)
        
        print(f"🔥 STRICT UNTOUCHED HOLDOUT IC: {holdout_ic:.4f} (p-value: {holdout_pval:.4f})")
        print("="*50)

    print("🏁 Serialized pipeline run successfully completed.")

