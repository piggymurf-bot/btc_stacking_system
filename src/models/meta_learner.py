import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

class DomainStackingClassifier:

    def __init__(self, base_pipelines: dict, meta_C: float = 0.1):
        self.base_pipelines = base_pipelines
        self.meta_learner = LogisticRegression(C=meta_C, solver='lbfgs', random_state=42)
        #self.model_names = list(base_pipelines.keys())

    def fit_stack(self, X: pd.DataFrame, y: pd.Series, tscv_n_splits = 5):
        """Generates base-model probabilities and fits the meta-learner."""
        
        inf_preds = pd.DataFrame(index=X.index, columns=self.base_pipelines.keys())
       
        tscv = TimeSeriesSplit(n_splits=tscv_n_splits)
        
        # 1. base-model probabilities Generation for Level-1
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            # Slice features and targets
            X_train_batch = X.iloc[train_index]
            y_train_batch = y.iloc[train_index]

            for name, spec in self.base_pipelines.items():
                model = spec['model']
                features = spec['features']

                # Fit base model ONLY on its designated feature group
                model.fit(X_train_batch[features], y_train_batch)
               
        for name, spec in self.base_pipelines.items():
            model = spec['model']
            features = spec['features']

            inf_preds.loc[X.index, name] = model.predict_proba(X[features])[:, 1]

        # 2. Fit Level-1 Meta-Learner on base-models probabilities
        self.meta_learner.fit(inf_preds, y)

        return self

    def predict_proba_stack(self, X: pd.DataFrame) -> np.ndarray:
        """Passes unseen features through base pipelines to meta-learner."""
        level_0_probs = pd.DataFrame(index=X.index, columns=self.base_pipelines.keys())

        for name, spec in self.base_pipelines.items():
            model = spec['model']
            features = spec['features']

            level_0_probs.loc[X.index, name] = model.predict_proba(X[features])[:, 1]

        return self.meta_learner.predict_proba(level_0_probs)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)