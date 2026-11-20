import pandas as pd
import numpy as np
import os
# import lightgbm as lgb
import joblib
# from sklearn.linear_model import Ridge
# from scipy.stats import spearmanr

from src.features.build_lhar_features import add_lhar_main
from src.features.build_alpha_targets import process_alpha_targets
from src.config_v2_mod import FEATURE_GROUPS
from src.models.v2_features_partition import determine_market_regime
from src.models.v2_alpha_engine import run_alpha_pipeline
from src.models.v2_volatility_engine import run_volat_pipeline


ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]
DERIVATIVE_FEATURES = FEATURE_GROUPS["derivatives_microstructure_features"]
TECHNICALS_FEATURES = FEATURE_GROUPS["technicals_and_volatility_features"]
all_features = ONCHAIN_FEATURES + DERIVATIVE_FEATURES + TECHNICALS_FEATURES

def calculate_psi(reference_series, current_series, num_bins=5):
    """Calculates Population Stability Index between reference and current feature distributions."""
    # Compute bin edges based on reference distribution quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(reference_series.dropna(), quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Get counts
    ref_counts = np.histogram(reference_series.dropna(), bins=bins)[0]
    curr_counts = np.histogram(current_series.dropna(), bins=bins)[0]
    
    # Convert to percentages (with small epsilon to avoid division by zero)
    ref_pct = (ref_counts / len(reference_series.dropna())) + 1e-4
    curr_pct = (curr_counts / len(current_series.dropna())) + 1e-4
    
    # Re-normalize percentages
    ref_pct /= ref_pct.sum()
    curr_pct /= curr_pct.sum()
    
    # PSI formula
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return psi

def holdout_df_generation_live(
        master_file: str = "data/processed/master_feature_matrix.csv",
        feature_cols: list[str] = all_features, 
        profile_dir_alpha: str = "data/regime_profiles_alpha",
        profile_dir_volat: str = "data/regime_profiles_volat",
        artifact_dir_alpha: str = "models/regime_stacked_alpha_artifacts",
        artifact_dir_volat: str = "models/regime_stacked_volat_artifacts",
        save_dir: str = "data/backtest"
    ):
    master_df = pd.read_csv(master_file, parse_dates=['date'], index_col='date')
   
    #Feature building block (without targets)
    df_features = add_lhar_main(
                        dir_path = "data/processed",
                        matrix_file = "master_feature_matrix.csv",
                    ) 
    
    df_features = df_features.set_index('date', drop=True)
    
    holdout_size = 600
    train_df = df_features.iloc[:-holdout_size].copy()      # Training set, used in training
    holdout_df = df_features.iloc[-holdout_size:].copy()    # Holdout set, used in backtest
    
    # Establish reference baseline from the last 30 days of training set
    garman_window_size = 90
    ref_gk = train_df['garman_klass_vol'].iloc[-garman_window_size:]
    psi_threshold = 0.40
    
    # Arrays that will be collected for holdout artifacts
    date_arr = []
    close_price = []
    psi_list = []
    regimes_alpha = []
    alpha_preds = []
    volat_preds = []
    actual_return = []
    garman_klass = []
    
    retrain_mode = False
    retrain_window = 60
    window_count = 0
    
    for date_idx in range(0,holdout_size):
        holdout_current = holdout_size - date_idx
        current_df = df_features.iloc[:-holdout_current].copy()
        target_df = holdout_df.iloc[[date_idx]].copy()
        
        date_arr.append(target_df.index[0])
    
        # Compute rolling PSI for Garman-Klass volatility for the last (window_size) days
        curr_window = current_df['garman_klass_vol'].iloc[-garman_window_size:]
        current_psi = calculate_psi(ref_gk, curr_window)        
        window_count += 1
        
        # if window_count > retrain_window:
        #     retrain_mode = True
        #     window_count = 0
        
        if retrain_mode:
       #if current_psi > psi_threshold and retrain_mode:
            # Severe drift detected, retrain the model
            print(f"Current psi: {current_psi} > {psi_threshold}. Triggering the training pipeline...")
            
            # Save the current matrix file and process targets            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'current_master_addlhar.csv')
            current_df.to_csv(save_path, index=True)
            
            process_alpha_targets(
                dir_path = save_dir,
                matrix_file = "current_master_addlhar.csv",
                output_file = "current_master_addlhar_alpha.csv",
                horizons = [1, 3, 7, 14]
            )
            
            # Training Alpha and Volatility engine
            current_master_path = os.path.join(save_dir, 'current_master_addlhar_alpha.csv')
            run_alpha_pipeline(
                master_file = current_master_path,
                holdout_size = 0,
                all_features = ONCHAIN_FEATURES + DERIVATIVE_FEATURES + TECHNICALS_FEATURES,
                target_col = 'target_fwd_cum_ret_14d'
            ) 
            
            run_volat_pipeline(
                master_file = current_master_path,
                holdout_size = 0,
                all_features = ONCHAIN_FEATURES + DERIVATIVE_FEATURES + TECHNICALS_FEATURES,
                target_col = 'target_rv_14d'
            )
            
            # Reset the Garman baseline 
            ref_gk = current_df['garman_klass_vol'].iloc[-garman_window_size:]
            psi_list.append(current_psi)
            
            retrain_mode = False
            
        else:
            # Drift begin within range, continue using the store models
            # print(f"Current psi: {current_psi} < {psi_threshold}. Continue...")
            psi_list.append(current_psi)
        
        
        # Determine the market regime for the target
        
        daily_df = pd.concat([current_df, target_df], ignore_index=False)
        
        window = 14
        slice_df = daily_df.iloc[-window:]
        regimes_alpha.append(determine_market_regime(slice_df, window=window))
        
        regimes = ['clean_trend', 'high_vol_chop', 'mean_reverting_sideways']
        
        # Evaluate Alpha Prediction
        alpha_df = daily_df.dropna(subset=['close'] + all_features).copy()

        def get_loaded_expert_matrix(df, loaded_experts):
            matrix = np.zeros((len(df), len(regimes)))
            for idx, regime in enumerate(regimes):
                if regime in loaded_experts:
                    exp = loaded_experts[regime]
                    X_sub = df[[f for f in exp["features"] if f in df.columns]]
                    matrix[:, idx] = exp["model"].predict(X_sub)
            return matrix

        # Load Alpha Artifacts and Predict
        loaded_experts_alpha = joblib.load(os.path.join(artifact_dir_alpha, "lastfold_expert_models.pkl"))
        loaded_meta_alpha = joblib.load(os.path.join(artifact_dir_alpha, "lastfold_meta_learner.pkl"))
    
        holdout_meta_X_alpha = get_loaded_expert_matrix(alpha_df, loaded_experts_alpha)
        holdout_preds_alpha = loaded_meta_alpha.predict(holdout_meta_X_alpha)
    
        alpha_preds.append(holdout_preds_alpha[-1])
        close_price.append(alpha_df['close'].iloc[-1])
        actual_return.append(alpha_df['close'].pct_change().iloc[-1])
        garman_klass.append(alpha_df['garman_klass_vol'].iloc[-1])
        
        # Evaluate Volatility Prediction
        volat_df = daily_df.dropna(subset=['close'] + all_features).copy()
  
        # Load Volatility Artifacts and Predict
        loaded_experts_volat = joblib.load(os.path.join(artifact_dir_volat, "lastfold_expert_models.pkl"))
        loaded_meta_volat = joblib.load(os.path.join(artifact_dir_volat, "lastfold_meta_learner.pkl"))
    
        holdout_meta_X_volat = get_loaded_expert_matrix(volat_df, loaded_experts_volat)
        holdout_preds_volat = loaded_meta_volat.predict(holdout_meta_X_volat)
        
        volat_preds.append(holdout_preds_volat[-1])

    # Join all collected arrays into Pandas Dataframe
    holdout_backtest = pd.DataFrame({
        'date' : date_arr,
        'close': close_price,
        'actual_return': actual_return,
        'market_regime': regimes_alpha,
        'pred_alpha': alpha_preds,
        'pred_volat': volat_preds,
        'psi': psi_list,
        'garman_klass_vol': garman_klass
    }).dropna()
    
    holdout_backtest = holdout_backtest.set_index('date',drop=True)
        
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
        
    holdout_df_generation_live(
            master_file = "data/processed/master_feature_matrix.csv",
            feature_cols = all_features, 
            profile_dir_alpha = "data/regime_profiles_alpha",
            profile_dir_volat = "data/regime_profiles_volat",
            artifact_dir_alpha = "models/regime_stacked_alpha_artifacts",
            artifact_dir_volat = "models/regime_stacked_volat_artifacts",
        )
    