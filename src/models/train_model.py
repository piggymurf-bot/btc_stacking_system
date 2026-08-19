import os
import pandas as pd
import numpy as np
from src.models.base_pipelines import build_base_pipelines
from src.models.meta_learner import DomainStackingClassifier


def run_training_pipeline():
    print('--- Loading Processed Data ---')
    
    # Accessing to saved data file
    file_path = os.path.join('data/processed', 'master_features_and_target.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Download and Process Data first.'
      )
    
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    #Splitting data in training set and holdout (test) set
    
    holdout_days = 0

    # Non-feature columns
    metadata_cols = [
        'date',
        'Close',
        'target',
        ] 
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Replace +inf and -inf with NaN, then forward-fill / backward-fill them
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
   
    # Temporal Split: Reserve last `holdout_days` for pure out-of-sample testing
    split_idx = len(df) - holdout_days

    X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]
        
    #Extract Close price for holdout dates
    close_df = pd.DataFrame()
    close_df['close'] = df.loc[X.index, 'Close']
      
    print('--- Building Base Pipelines ---')
    base_pipelines = build_base_pipelines()

    print('--- Fitting Level-0 and Level-1 Meta-Learner ---')
    stacker = DomainStackingClassifier(base_pipelines, meta_C=0.1)
    oof_meta_preds = stacker.fit_stack(X_train, y_train)

    # Save trained model artifact
    os.makedirs('models', exist_ok=True)
    stacker.save('models/stacking_ensemble.pkl')
    print('Saved ensemble to models/stacking_ensemble.pkl')

    # Generate Meta-learner Predictions for Phase 4 (Backtesting)
    #holdout_probs = stacker.predict_proba_stack(X_holdout)[:, 1] # Use this line if want to use holdout dates for backtest, be sure that "holdout_days" is not 0
    holdout_probs = oof_meta_preds['meta_ypreds'] # Use this line if want to use out-of-fold prediction generated during the fitting for walk-forward backtest
    holdout_df = pd.DataFrame(
        {
            "y_true": y,
            "y_pred_prob": holdout_probs,
            "close": close_df["close"],
            },
        #index=X_holdout.index,
        index = oof_meta_preds.index,
    )

    holdout_df.to_csv("data/processed/holdout_predictions.csv")

    print('Saved predictions to data/processed/holdout_predictions.csv')


if __name__ == '__main__':
  run_training_pipeline()