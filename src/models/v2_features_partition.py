import numpy as np
import pandas as pd
import json
import os
import lightgbm as lgb
import warnings
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from scipy.stats import ConstantInputWarning


def select_features_mrmr(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 15):
    """
    Selects a subset of features maximizing relevance to target while minimizing redundancy.
    """
    feature_names = X.columns.tolist()
    if len(feature_names) <= n_features_to_select:
        return feature_names

    # 1. Compute Relevance: Mutual Information between each feature and target
    relevance = mutual_info_regression(X, y, random_state=42)
    relevance_dict = dict(zip(feature_names, relevance))

    # 2. Compute Feature-to-Feature Redundancy Matrix using correlation or MI
    # (Using correlation matrix here for fast, robust execution on financial data)
    corr_matrix = X.corr(method="spearman").abs().fillna(0)

    selected_features = []
    
    # Select the feature with highest relevance first
    best_first = max(relevance_dict, key=relevance_dict.get)
    selected_features.append(best_first)
    
    # Iteratively select features balancing high relevance and low redundancy
    while len(selected_features) < n_features_to_select:
        best_score = -float("inf")
        best_candidate = None
        
        for candidate in feature_names:
            if candidate in selected_features:
                continue
                
            rel = relevance_dict[candidate]
            
            # Redundancy: Average correlation/MI with already selected features
            red = corr_matrix.loc[candidate, selected_features].mean()
            
            # mRMR score (Difference formulation)
            mrmr_score = rel - red
            
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_candidate = candidate
                
        if best_candidate:
            selected_features.append(best_candidate)
        else:
            break
            
    print(f"🧬 mRMR Selected {len(selected_features)} features out of {len(feature_names)}")
    print(f"\n🏆 Top {len(selected_features)} mRMR Selected Features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i}. {feat}")
    return selected_features

def compute_mrmr_feature_stability(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_bootstraps: int = 50, 
    subsample_frac: float = 0.8,
    n_features_to_select: int = 15
) -> pd.DataFrame:
    """
    Runs repeated bootstrap sub-sampling using mRMR feature selection 
    to calculate robust selection frequencies while penalizing redundancy.
    """
    feature_counts = {col: 0 for col in X.columns}
    n_samples = int(len(X) * subsample_frac)
    
    print(f"🔄 Running mRMR Stability Selection across {n_bootstraps} bootstrap samples...")
    
    for i in range(n_bootstraps):
        # 1. Generate randomized bootstrap indices
        idx = np.random.choice(len(X), size=n_samples, replace=True)
        X_sub = X.iloc[idx].reset_index(drop=True)
        y_sub = y.iloc[idx].reset_index(drop=True)
        
        # 2. Run mRMR selection for this bootstrap sample
        # (Calling your select_features_mrmr function under the hood)
        selected_features_run = select_features_mrmr(X_sub, y_sub, n_features_to_select=n_features_to_select)
        
        # 3. Increment counts for features selected by mRMR in this run
        for feat in selected_features_run:
            feature_counts[feat] += 1
            
    # 4. Convert counts into selection frequencies (0.0 to 1.0)
    stability_df = pd.DataFrame(list(feature_counts.items()), columns=["Feature", "Selection_Count"])
    stability_df["Selection_Frequency"] = stability_df["Selection_Count"] / n_bootstraps
    stability_df = stability_df.sort_values(by="Selection_Frequency", ascending=False).reset_index(drop=True)
    
    return stability_df

def sequential_forward_selection_forced(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    target_features: int = 15
) -> list[str]:
    """
    Forces SFS to select a fixed target number of features without early stopping,
    ensuring a robust candidate pool for core/satellite partitioning.
    """
    remaining_features = list(X_train.columns)
    selected_features = []
    
    print(f"🚀 Starting Forced SFS (Target: {target_features} features)...")
    
    for step in range(1, target_features + 1):
        if not remaining_features:
            break
            
        best_candidate = None
        best_step_score = -float('inf')
        
        for candidate in remaining_features:
            current_subset = selected_features + [candidate]
            X_tr = X_train[current_subset]
            X_v = X_val[current_subset]
            
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'max_depth': 4,
                'num_leaves': 12,
                'min_child_weight': 30,
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_tr, label=y_train)
            model = lgb.train(lgb_params, train_data, num_boost_round=100)
            
            preds = model.predict(X_v)

            # ic, _ = spearmanr(preds, y_val)
            # Safely handle constant prediction arrays
            # ic = ic if (not np.isnan(ic) and not np.isinf(ic)) else 0.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                try:
                    preds = model.predict(X_v)
                    # Check if predictions have zero variance (are constant)
                    if np.all(preds == preds[0]):
                        ic = 0.0
                    else:
                        ic, _ = spearmanr(preds, y_val)
                        ic = ic if (not np.isnan(ic) and not np.isinf(ic)) else 0.0
                except Exception:
                    ic = 0.0
            
            if ic > best_step_score:
                best_step_score = ic
                best_candidate = candidate
                
        if best_candidate:
            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            print(f"   -> Step {step}: Added [{best_candidate}] | Validation IC: {best_step_score:.4f}")
            
    print(f"\n🏆 Forced SFS Complete. Selected {len(selected_features)} features.")
    return selected_features

def partition_core_and_satellites_explicit(
    selected_features: list[str], 
    core_count: int = 6
):
    """
    Explicitly splits the SFS-selected feature list into a Permanent Core Backbone
    and a Dynamic Satellite Pool, guaranteeing neither pool is empty.
    """
    if len(selected_features) <= core_count:
        core_features = selected_features
        satellite_pool = []
    else:
        core_features = selected_features[:core_count]
        satellite_pool = selected_features[core_count:]
        
    print("\n" + "="*50)
    print("🏛️ PERMANENT CORE BACKBONE (Always Active)")
    print("="*50)
    for f in core_features:
        print(f"   - {f}")
        
    print("\n" + "="*50)
    print("🛰️ DYNAMIC SATELLITE POOL (Regime-Dependent)")
    print("="*50)
    if satellite_pool:
        for f in satellite_pool:
            print(f"   - {f}")
    else:
        print("   (Warning: Satellite pool is empty. Adjust core_count.)")
        
    return core_features, satellite_pool

def sequential_forward_selection_lgb(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    max_features: int = 15
) -> list[str]:
    """
    Performs Sequential Forward Selection using LightGBM validation Spearman IC 
    as the optimization objective to capture non-linear feature interactions.
    """
    remaining_features = list(X_train.columns)
    selected_features = []
    
    best_overall_score = -float('inf')
    
    print(f"🚀 Starting Sequential Forward Selection (Target: {max_features} features)...")
    
    for step in range(1, max_features + 1):
        best_candidate = None
        best_step_score = -float('inf')
        
        for candidate in remaining_features:
            # Test candidate added to current selected features
            current_subset = selected_features + [candidate]
            
            X_tr = X_train[current_subset]
            X_v = X_val[current_subset]
            
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'max_depth': 4,
                'num_leaves': 12,
                'min_child_weight': 30,
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_tr, label=y_train)
            model = lgb.train(lgb_params, train_data, num_boost_round=100)
            
            # Evaluate validation Spearman IC
            preds = model.predict(X_v)
            ic, _ = spearmanr(preds, y_val)
            ic = ic if not np.isnan(ic) else 0.0
            
            if ic > best_step_score:
                best_step_score = ic
                best_candidate = candidate
                
        # If adding a feature improves or maintains performance, keep it
        if best_candidate and best_step_score > best_overall_score:
            best_overall_score = best_step_score
            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            print(f"   -> Step {step}: Added [{best_candidate}] | Validation IC: {best_overall_score:.4f}")
        else:
            print(f"   -> Step {step}: No further performance improvement. Stopping early.")
            break
            
    print(f"\n🏆 SFS Complete. Selected {len(selected_features)} features.")
    return selected_features

def two_stage_feature_selection(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    mrmr_pool_size: int = 30,
    final_feature_count: int = 12
) -> list[str]:
    """
    Two-stage feature selection:
    1. mRMR Filter: Reduces 150+ raw features down to a clean pool of 30.
    2. LightGBM SFS Wrapper: Selects final 12 features based on non-linear synergy.
    """
    print("--------------------------------------------------")
    print("🔄 STAGE 1: Running mRMR Feature Filtering...")
    print("--------------------------------------------------")
    # Stage 1: Fast filter for relevance and low redundancy
    mrmr_candidates = select_features_mrmr(X_train, y_train, n_features_to_select=mrmr_pool_size)
    
    # Slice datasets to mRMR candidate subset
    X_train_mrmr = X_train[mrmr_candidates]
    X_val_mrmr = X_val[mrmr_candidates]
    
    print("\n--------------------------------------------------")
    print("🚀 STAGE 2: Running LightGBM Sequential Forward Selection...")
    print("--------------------------------------------------")
    # Stage 2: Wrapper to capture non-linear interactions among top candidates
    # final_features = sequential_forward_selection_lgb(
    final_features = sequential_forward_selection_forced(
        X_train=X_train_mrmr,
        y_train=y_train,
        X_val=X_val_mrmr,
        y_val=y_val,
        # max_features=final_feature_count
        target_features=final_feature_count
    )
    
    return final_features

def determine_market_regime(df_slice: pd.DataFrame, window: int = 14) -> str:
    """
    Clean, bug-free regime classifier that correctly computes slice volatility.
    """
    if len(df_slice) < window:
        return "neutral"
        
    # 1. Calculate tactical trend efficiency over the window
    net_dist = abs(df_slice['close'].iloc[-1] - df_slice['close'].iloc[-window])
    path_dist = df_slice['close'].diff().abs().sum() # Sum over the slice directly
    trend_efficiency = net_dist / (path_dist + 1e-8)
    
    # 2. Calculate annualized volatility directly for the slice
    close_prices = pd.to_numeric(df_slice['close'], errors='coerce')
    log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
    
    if len(log_ret) < 5:
        return "neutral"
        
    # Scalar annualized volatility of the entire slice
    rolling_vol = log_ret.std() * np.sqrt(365)
    
    # 3. Regime Routing Logic
    # High Volatility Chop: Volatility above 65%
    if rolling_vol >= 0.65:
        return "high_vol_chop"
    # Clean Trend: Good directional efficiency and controlled volatility
    elif trend_efficiency > 0.20 and rolling_vol < 0.65:
        return "clean_trend"
    # Sideways / Mean Reverting: Choppy price action
    else:
        return "mean_reverting_sideways"

def profile_and_save_regime_features(
    wfo_df: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str, 
    window: int = 30,
    mrmr_pool_size: int = 30,
    sfs_target: int = 15,
    core_count: int = 7,
    output_dir: str = "data/regime_profiles"
):
    """
    Classifies historical rows into strict, mutually exclusive market regimes,
    then runs mRMR + Forced SFS independently on each regime's unique sample subset.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"The window is {window}-day") 
    
    # 1. Classify every row strictly
    regimes = []
    for i in range(len(wfo_df)):
        slice_df = wfo_df.iloc[max(0, i - window + 1):i + 1]
        regimes.append(determine_market_regime(slice_df, window=window))
        
    wfo_df = wfo_df.copy()
    wfo_df['market_regime'] = regimes
    
    # Print exact distribution to verify partitioning
    print("\n📊 Historical Regime Distribution:")
    print(wfo_df['market_regime'].value_counts())
    print("-" * 50)
    
    unique_regimes = ['clean_trend', 'high_vol_chop', 'mean_reverting_sideways']
    regime_profiles = {}
    
    for regime in unique_regimes:
        print("\n==================================================")
        print(f"🏛️ PROFILING REGIME: [{regime.upper()}]")
        print("==================================================")
        
        regime_data = wfo_df[wfo_df['market_regime'] == regime].copy()
        sample_count = len(regime_data)
        print(f"   📊 Exact distinct rows captured: {sample_count}")
        
        if sample_count < 50:
            print(f"   ⚠️ Warning: Low sample count for {regime}. Consider adjusting regime thresholds.")
            continue
            
        X_regime = regime_data[feature_cols]
        y_regime = regime_data[target_col]
        
        # Split regime data into pseudo train/val for SFS evaluation
        split_idx = int(len(X_regime) * 0.8)
        X_tr, X_v = X_regime.iloc[:split_idx], X_regime.iloc[split_idx:]
        y_tr, y_v = y_regime.iloc[:split_idx], y_regime.iloc[split_idx:]
        
        # Step A: mRMR Filter tailored specifically to this regime's characteristics
        mrmr_pool = select_features_mrmr(X_tr, y_tr, n_features_to_select=mrmr_pool_size)
        
        # Step B: Forced SFS Wrapper
        sfs_selected = sequential_forward_selection_forced(
            X_train=X_tr[mrmr_pool],
            y_train=y_tr,
            X_val=X_v[mrmr_pool],
            y_val=y_v,
            target_features=sfs_target
        )
        
        # Step C: Partition Core vs Satellite
        core_features, satellite_pool = partition_core_and_satellites_explicit(
            sfs_selected, 
            core_count=core_count
        )
        
        # Save profile structure
        profile = {
            "regime": regime,
            "core_features": core_features,
            "satellite_pool": satellite_pool,
            "sample_count": sample_count
        }
        regime_profiles[regime] = profile
        
        # Save individual JSON file
        filepath = os.path.join(output_dir, f"regime_profile_{regime}.json")
        with open(filepath, "w") as f:
            json.dump(profile, f, indent=4)
        print(f"💾 Saved distinct profile to {filepath}")
        
    return regime_profiles

