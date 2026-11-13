import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import joblib
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

from src.config_v2_mod import FEATURE_GROUPS
from src.models.v2_features_partition import determine_market_regime

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]
DERIVATIVE_FEATURES = FEATURE_GROUPS["derivatives_microstructure_features"]
TECHNICALS_FEATURES = FEATURE_GROUPS["technicals_and_volatility_features"]
all_features = ONCHAIN_FEATURES + DERIVATIVE_FEATURES + TECHNICALS_FEATURES

def holdout_df_generation(
        master_file: str = "data/processed/master_feature_addlhar_alpha.csv",
        feature_cols: list[str] = all_features, 
        target_col_alpha: str = "target_fwd_cum_ret_14d",
        target_col_volat: str = "target_rv_14d",
        holdout_size: int = 150,
        profile_dir_alpha: str = "data/regime_profiles_alpha",
        profile_dir_volat: str = "data/regime_profiles_volat",
        artifact_dir_alpha: str = "models/regime_stacked_alpha_artifacts",
        artifact_dir_volat: str = "models/regime_stacked_volat_artifacts",
        save_dir: str = "data/backtest"
    ):
    master_df = pd.read_csv(master_file, parse_dates=['date'], index_col='date')
    
    # --- 1. Evaluate Alpha Holdout (matching training slice logic) ---
    alpha_clean = master_df.dropna(subset=[target_col_alpha, 'close'] + all_features)
    holdout_alpha_df = alpha_clean.iloc[-holdout_size:].copy()
    
    # Compute regimes for alpha holdout
    regimes_alpha = []
    window = 14
    for i in range(len(holdout_alpha_df)):
        slice_df = holdout_alpha_df.iloc[max(0, i - window + 1):i + 1]
        regimes_alpha.append(determine_market_regime(slice_df, window=window))
    holdout_alpha_df['market_regime'] = regimes_alpha
    
    regimes = ['clean_trend', 'high_vol_chop', 'mean_reverting_sideways']

    def get_loaded_expert_matrix(df, loaded_experts):
        matrix = np.zeros((len(df), len(regimes)))
        for idx, regime in enumerate(regimes):
            if regime in loaded_experts:
                exp = loaded_experts[regime]
                X_sub = df[[f for f in exp["features"] if f in df.columns]]
                matrix[:, idx] = exp["model"].predict(X_sub)
        return matrix

    # Load Alpha Artifacts and Predict
    loaded_experts_alpha = joblib.load(os.path.join(artifact_dir_alpha, "fold3_expert_models.pkl"))
    loaded_meta_alpha = joblib.load(os.path.join(artifact_dir_alpha, "fold3_meta_learner.pkl"))
    
    holdout_meta_X_alpha = get_loaded_expert_matrix(holdout_alpha_df, loaded_experts_alpha)
    holdout_preds_alpha = loaded_meta_alpha.predict(holdout_meta_X_alpha)
    
    holdout_alpha_df['pred_alpha'] = holdout_preds_alpha
    holdout_alpha_df['actual_return'] = holdout_alpha_df['close'].pct_change()
    
    holdout_ic_a, holdout_pval_a = spearmanr(holdout_preds_alpha, holdout_alpha_df[target_col_alpha])
    print(f"🔥 HOLDOUT IC (Alpha): {holdout_ic_a:.4f} (p-value: {holdout_pval_a:.4f})")
    print("="*50)
    
    # --- 2. Evaluate Volatility Holdout (independent cleaning slice) ---
    volat_clean = master_df.dropna(subset=[target_col_volat, 'close'] + all_features)
    holdout_volat_df = volat_clean.iloc[-holdout_size:].copy()
    
    regimes_volat = []
    for i in range(len(holdout_volat_df)):
        slice_df = holdout_volat_df.iloc[max(0, i - window + 1):i + 1]
        regimes_volat.append(determine_market_regime(slice_df, window=window))
    holdout_volat_df['market_regime'] = regimes_volat
    
    loaded_experts_volat = joblib.load(os.path.join(artifact_dir_volat, "fold3_expert_models.pkl"))
    loaded_meta_volat = joblib.load(os.path.join(artifact_dir_volat, "fold3_meta_learner.pkl"))
    
    holdout_meta_X_volat = get_loaded_expert_matrix(holdout_volat_df, loaded_experts_volat)
    holdout_preds_volat = loaded_meta_volat.predict(holdout_meta_X_volat)
    
    holdout_ic_v, holdout_pval_v = spearmanr(holdout_preds_volat, holdout_volat_df[target_col_volat])
    print(f"🔥 HOLDOUT IC (Volatility): {holdout_ic_v:.4f} (p-value: {holdout_pval_v:.4f})")
    print("="*50)
    
    # --- 3. Merge Backtest DataFrame on Date Index ---
    holdout_backtest = pd.DataFrame({
        'close': holdout_alpha_df['close'],
        'actual_return': holdout_alpha_df['actual_return'],
        'market_regime': holdout_alpha_df['market_regime'],
        'pred_alpha': holdout_alpha_df['pred_alpha'],
        'pred_volat': pd.Series(holdout_preds_volat, index=holdout_volat_df.index)
    }).dropna()
    
    # Print exact distribution to verify partitioning
    print("\n📊 Historical Regime Distribution:")
    print(holdout_backtest['market_regime'].value_counts())
    print("-" * 50)
    
    print(holdout_backtest.head())
    print(holdout_backtest.tail())
    
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'holdout_backtest.csv')
    holdout_backtest.to_csv(save_path, index=True)
    
    return holdout_backtest
    
if __name__ == '__main__':
        
    holdout_df_generation(
            master_file = "data/processed/master_feature_addlhar_alpha.csv",
            feature_cols = all_features, 
            target_col_alpha = "target_fwd_cum_ret_14d",
            target_col_volat = "target_rv_14d",
            holdout_size = 150,
            profile_dir_alpha = "data/regime_profiles_alpha",
            profile_dir_volat = "data/regime_profiles_volat",
            artifact_dir_alpha = "models/regime_stacked_alpha_artifacts",
            artifact_dir_volat = "models/regime_stacked_volat_artifacts",
        )
    