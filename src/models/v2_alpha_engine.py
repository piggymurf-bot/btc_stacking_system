import numpy as np
import pandas as pd
import os

from src.config_v2_mod import FEATURE_GROUPS
from src.models.v2_features_partition import profile_and_save_regime_features
from src.models.v2_optuna_tuning import run_optuna_pipeline
from src.models.v2_wfo_training import run_regime_stacked_serialized_pipeline

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]
DERIVATIVE_FEATURES = FEATURE_GROUPS["derivatives_microstructure_features"]
TECHNICALS_FEATURES = FEATURE_GROUPS["technicals_and_volatility_features"]

def run_alpha_pipeline(
     master_file: str = "data/processed/master_feature_addlhar_alpha.csv",
     holdout_size: int = 150,
     all_features: list[str] = ONCHAIN_FEATURES + DERIVATIVE_FEATURES + TECHNICALS_FEATURES,
     target_col: str = 'target_fwd_cum_ret_14d'
):
    
    master_df = pd.read_csv(master_file, parse_dates=['date'], index_col='date')
    master_df_clean = master_df.dropna(subset=[target_col, 'close'] + all_features)
    holdout_df = master_df_clean.iloc[-holdout_size:].copy()
    wfo_df = master_df_clean.iloc[:-holdout_size].copy()

    profile_and_save_regime_features(
        wfo_df = wfo_df, 
        feature_cols = all_features, 
        target_col = target_col, 
        window = 14,
        mrmr_pool_size = 30,
        sfs_target = 15,
        core_count = 7,
        output_dir = "data/regime_profiles_alpha"
    )
    
    best_params = run_optuna_pipeline(
                            wfo_df = wfo_df,
                            feature_cols = all_features,
                            target_col = target_col,
                            study_name="bitcoin_alpha_study",
                            db_path="sqlite:///data/optuna_db/optuna_alpha_study.db",
                            profile_dir="data/regime_profiles_alpha",
                            n_trials=30
                    )
    
    # 1. Extract the meta-learner parameter
    best_meta_alpha = best_params.get('meta_alpha', 1.0)

    # 2. Extract tuned LightGBM hyperparameters dynamically
    lgb_tuned_keys = [
        'learning_rate', 'max_depth', 'num_leaves', 'min_child_weight', 
        'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample'
    ]
    best_lgb_params = {k: best_params[k] for k in lgb_tuned_keys if k in best_params}

    # 3. Re-inject your fixed structural configurations for LightGBM
    best_lgb_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42
    })

    print("--- Tuned LGB Params ---")
    print(best_lgb_params)
    print(f"\n--- Tuned Meta Alpha: {best_meta_alpha} ---")
        
    run_regime_stacked_serialized_pipeline(
        master_df = master_df, 
        feature_cols = all_features, 
        target_col = target_col, 
        lgb_params = best_lgb_params,
        meta_alpha = best_meta_alpha,
        val_block_size = 246,
        holdout_size = holdout_size,
        profile_dir = "data/regime_profiles_alpha",
        artifact_dir = "models/regime_stacked_alpha_artifacts",
        experiment_name = "Bitcoin_Regime_Stacked_Serialized_alpha",
    )
    
    
if __name__ == '__main__':
    
    run_alpha_pipeline(
        master_file = "data/processed/master_feature_addlhar_alpha.csv",
        holdout_size = 150,
        all_features = ONCHAIN_FEATURES + DERIVATIVE_FEATURES + TECHNICALS_FEATURES,
        target_col = 'target_fwd_cum_ret_14d'
    ) 