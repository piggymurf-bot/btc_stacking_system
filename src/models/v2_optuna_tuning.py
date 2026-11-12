import optuna
import optuna.visualization as vis
import random
from optuna.samplers import TPESampler
import json
import os
import mlflow
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


from src.models.v2_features_partition import determine_market_regime

def objective(trial, wfo_df, feature_cols, target_col, profile_dir="data/regime_profiles"):
    
    # 1. Start a nested MLflow run for each individual Optuna trial
    with mlflow.start_run(nested=True):
        
        # 1. Sample hyperparameters for LightGBM base experts
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'num_leaves': trial.suggest_int('num_leaves', 8, 24),
            'min_child_weight': trial.suggest_int('min_child_weight', 30, 80),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 5.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'verbose': -1,
            'random_state': 42
        }
        
        # 2. Sample regularization for Ridge meta-learner
        meta_alpha = trial.suggest_float('meta_alpha', 0.1, 10.0, log=True)
        
        # Log all trial parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("meta_alpha", meta_alpha)
        mlflow.log_param("target_col", target_col)
        
        val_block_size = 246
        total_rows = len(wfo_df)
        horizon = int(target_col.split('_')[-1].replace('d', ''))

        regimes = []
        window = 14
        for i in range(len(wfo_df)):
            slice_df = wfo_df.iloc[max(0, i - window + 1):i + 1]
            regimes.append(determine_market_regime(slice_df, window=window))
            
        wfo_df = wfo_df.copy()
        wfo_df['market_regime'] = regimes
        
        regimes = ['clean_trend', 'high_vol_chop', 'mean_reverting_sideways']
        
        oof_meta_preds = []
        oof_actuals = []
        
        # 3. WFO Fold Cross-Validation Loop
        for fold_idx in range(1, 4):
            train_end_idx = total_rows - (4 - fold_idx) * val_block_size - horizon
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + val_block_size, total_rows)
            
            train_df = wfo_df.iloc[:train_end_idx].copy()
            val_df = wfo_df.iloc[val_start_idx:val_end_idx].copy()
            
            if len(val_df) < 30:
                mlflow.log_metric("aggregate_oof_ic", -999.0)
                return -999.0
                
            y_train = train_df[target_col]
            y_val = val_df[target_col]
            
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
                model = lgb.train(params, lgb_train, num_boost_round=100)
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
            
            meta_learner = Ridge(alpha=meta_alpha)
            meta_learner.fit(train_meta_X, y_train)
            
            meta_preds = meta_learner.predict(val_meta_X)
            oof_meta_preds.extend(meta_preds)
            oof_actuals.extend(y_val.values)
            
        aggregate_ic, _ = spearmanr(oof_meta_preds, oof_actuals)
        final_score = aggregate_ic if not np.isnan(aggregate_ic) else -999.0
        
        # Log the final evaluation metric to MLflow for this trial
        mlflow.log_metric("aggregate_oof_ic", final_score)
        
        return final_score

# Execute Optuna Study for Alpha or Volatility Engine
def run_optuna_pipeline(
    wfo_df, 
    feature_cols, 
    target_col, 
    study_name: str = "bitcoin_study", 
    db_path: str = "sqlite:///data/optuna_db/optuna_bitcoin_study.db",
    profile_dir: str = "data/regime_profiles", 
    n_trials: int = 30,
    custom_seed: int = None
):
    # Ensure your target directory exists
    os.makedirs("data/optuna_db", exist_ok=True)
    os.makedirs("data/ml_db", exist_ok=True)
    
    mlflow.set_tracking_uri("sqlite:///data/ml_db/mlflow.db")
    # Dynamically separate MLflow experiments per target
    mlflow.set_experiment(f"Bitcoin_Regime_Optuna_{target_col}")
    
    if custom_seed is None:
        run_seed = random.randint(0, 100000)
    else:
        run_seed = custom_seed
        
    print(f"🎲 Initializing Optuna Study with Seed: {run_seed}")
    sampler = TPESampler(seed=run_seed)
    
    # Create study with dynamic persistent storage and study name
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,  # Safe to re-run if study already exists
        direction="maximize",
        sampler=sampler
    )
    
    # Optional: Log the study-level seed to MLflow
    with mlflow.start_run(run_name=f"study_run_{target_col}"):
        mlflow.log_param("optuna_seed", run_seed)
        mlflow.log_param("study_name", study_name)
    
    study.optimize(
        lambda trial: objective(trial, wfo_df, feature_cols, target_col, profile_dir), 
        n_trials=n_trials, 
    )
    
    print("=" * 60)
    print(f"🏆 OPTUNA TUNING COMPLETE FOR: {target_col}")
    print(f"Best Score: {study.best_value:.4f}")
    print("Best Parameters:", study.best_params)
    print("=" * 60)
    
    return study.best_params

# Visualizing the Study
def optuna_visualize_study(study_name: str, study_path: str):
    
    # Reload the completed study from disk
    storage_name = study_path
    study = optuna.load_study(
        study_name = study_name,
        storage=storage_name
    )

    fig_history = vis.plot_optimization_history(study)
    fig_history.show()

    fig_importances = vis.plot_param_importances(study)
    fig_importances.show()
