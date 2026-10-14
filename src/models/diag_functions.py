import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def evaluate_predictions(y_true, y_pred):
    """
    Evaluates predictions using Rank IC (Spearman), Pearson IC, and RMSE.
    """
    # Drop NaNs if any
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid_mask], y_pred[valid_mask]
    
    # Calculate metrics
    spearman_ic, _ = spearmanr(yt, yp)
    pearson_ic, _ = pearsonr(yt, yp)
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    
    # Directional Hit Rate (Sign Accuracy)
    sign_match = (np.sign(yt) == np.sign(yp)).mean()
    
    print(f"📊 Spearman Rank IC : {spearman_ic:.4f}")
    print(f"📊 Pearson IC       : {pearson_ic:.4f}")
    print(f"📊 Directional Acc  : {sign_match * 100:.2f}%")
    print(f"📊 RMSE             : {rmse:.5f}")

# Example usage after generating OOF or validation predictions:
# evaluate_predictions(y_val, val_preds)

def diagnostic_feature_signal_check(X, y):
    print("🔍 Running Individual Feature IC Diagnostic...")
    ic_results = {}
    for col in X.columns:
        # Drop NaNs for calculation
        valid_idx = ~(X[col].isna() | y.isna())
        if valid_idx.sum() > 100:
            corr, pval = spearmanr(X[col][valid_idx], y[valid_idx])
            ic_results[col] = corr
            
    ic_df = pd.DataFrame(list(ic_results.items()), columns=["Feature", "Spearman_IC"]).sort_values(by="Spearman_IC", key=abs, ascending=False)
    print("\n--- TOP 10 FEATURES BY INDIVIDUAL IC ---")
    print(ic_df.head(10))
    
    print("\n🔍 Running Target Permutation Test (Sanity Check)...")
    # Shuffle y to destroy real signal
    y_shuffled = np.random.permutation(y)
    corr_shuffled, _ = spearmanr(X.iloc[:, 0], y_shuffled) # Quick check on first feature
    print(f"Feature-to-Random-Target IC (Should be ~0.0): {corr_shuffled:.4f}")

# Usage:
# diagnostic_feature_signal_check(X_train, y_train)