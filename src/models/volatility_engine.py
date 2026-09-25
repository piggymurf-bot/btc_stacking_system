import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROCESSED_DIR = "data/processed"

def train_lhar_model(
    df_master: pd.DataFrame, 
    start_date: str = "2023-01-01",
    include_microstructure: bool = True,
    include_leverage: bool = True
):
    """Fits an OLS Leveraged HAR (LHAR-RV-X) model with Newey-West HAC robust standard errors.

    """
    print(f"⚡ Training LHAR-RV-X Volatility Engine from {start_date}...")
    
    df = df_master[df_master["date"] >= start_date].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Define base HAR predictors
    feature_cols = ["rv_d", "rv_w", "rv_m"]

    # Incorporate Leverage / Asymmetric Semivariance Terms
    if include_leverage:
        leverage_cols = ["rv_neg_d", "rv_pos_d", "rv_neg_w", "rv_pos_w"]
        available_lev = [col for col in leverage_cols if col in df.columns]
        feature_cols.extend(available_lev)

    # Incorporate Microstructure & Implied Volatility (LHAR-RV-X)
    if include_microstructure:
        extended_features = [
            "obi_mean_daily", 
            "obi_std_daily", 
            "book_depth_1m_mean", 
            "bid_ask_depth_ratio",
            "dvol_close"
        ]
        available_ext = [col for col in extended_features if col in df.columns]
        feature_cols.extend(available_ext)

    # Clean missing values resulting from rolling windows and target shift
    df_model = df.dropna(subset=feature_cols + ["target_rv"]).copy()

    X = df_model[feature_cols]
    y = df_model["target_rv"]

    # Add constant intercept
    X_const = sm.add_constant(X)

    # Fit OLS with Newey-West HAC covariance (5 daily lags)
    model = sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # Model Evaluation Metrics
    y_pred = model.predict(X_const)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\n========================================================")
    print("               LHAR-RV-X MODEL RESULTS                  ")
    print("========================================================")
    print(model.summary())
    print("\n📊 Model Diagnostics & Performance Metrics:")
    print(f" - R-Squared (R²): {r2:.4f}")
    print(f" - RMSE:           {rmse:.6f}")
    print(f" - MAE:            {mae:.6f}")
    print("========================================================")

    df_model["predicted_rv"] = y_pred
    return model, df_model

if __name__ == "__main__":
    master_path = os.path.join(PROCESSED_DIR, "master_feature_matrix.csv")
    
    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path)
        model, df_results = train_lhar_model(
            df_master, 
            start_date="2023-01-01", 
            include_microstructure=True,
            include_leverage=True
        )
    else:
        print(f"❌ Could not find {master_path}. Run master_pipeline.py first.")