from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


from src.config import FEATURE_GROUPS


def build_base_pipelines():
    """Constructs domain-restricted base Pipelines for Level-0."""
    onchain_pipe = make_pipeline(
                        StandardScaler(), 
                        PCA(n_components=None),
                        XGBClassifier(
                            n_estimators=100,
                            max_depth=3,  # Keep trees shallow (depth 2-3) to learn macro rules, not micro noise
                            learning_rate=0.02,  # Slow learning rate for stable gradient convergence
                            subsample=0.8,  # Train each tree on 80% of rows
                            colsample_bytree=0.55,  # Force each split to see only 50% of on-chain features
                            reg_alpha=0.1,  # L1 regularization to prune redundant UTXO age bands
                            reg_lambda=1.0,  # L2 regularization to prevent extreme leaf weights
                            random_state=42,
                        )
                    )

    derivatives_pipe = make_pipeline(
                        StandardScaler(), 
                        PCA(n_components=None),
                        LogisticRegression(C= 0.1, solver='lbfgs', random_state=42)
                    )

    technicals_pipe = make_pipeline(
                        StandardScaler(), 
                        PCA(n_components=None),
                        LogisticRegression(C= 0.1, solver='lbfgs', random_state=42)
                    )


    # Define specialized base models mapped to their feature groups
    base_pipelines = {
                    "onchain_model": {
                                        "model": onchain_pipe,
                                        "features": FEATURE_GROUPS["onchain_features"],
                    },
                    "derivatives_model": {
                                        "model": derivatives_pipe,
                                        "features": FEATURE_GROUPS["derivatives_features"],
                    },
                    "technicals_model": {
                                        "model": technicals_pipe,
                                        "features": FEATURE_GROUPS["technicals_features"],
                    },
    }
    
    """
    return {
      'onchain_model': onchain_pipe,
      'derivatives_model': derivatives_pipe,
      'technicals_model': technicals_pipe,
    }
    """
    return (base_pipelines)
    