import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

# Set visual style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports/figures"

def evaluate_and_plot_volatility(
    df_results: pd.DataFrame, 
    output_path: str = os.path.join(REPORTS_DIR, "har_rv_x_diagnostics.png")
):
    """Generates a 4-panel visual diagnostic dashboard for the HAR-RV-X model:

    1. Time Series: Actual vs. Predicted Realized Volatility
    2. Prediction Errors: Residual Distribution & Density
    3. Scatter & Fit: Actual vs. Predicted Target Calibration
    4. Residual Autocorrelation (ACF): Checking for Remaining Serial Dependence
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    df = df_results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Compute Residuals & Annualized Values (for cleaner axis displays in % terms)
    # Annualized Volatility % = Volatility * sqrt(365) * 100
    ann_factor = np.sqrt(365) * 100
    df["actual_ann"] = df["target_rv"] * ann_factor
    df["pred_ann"] = df["predicted_rv"] * ann_factor
    df["residuals_ann"] = df["actual_ann"] - df["pred_ann"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    fig.suptitle("HAR-RV-X Model Diagnostics & Volatility Forecast Performance", fontsize=16, fontweight="bold", y=0.98)

    # ----------------------------------------------------
    # Panel 1: Time Series - Actual vs Predicted
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(df["date"], df["actual_ann"], label="Actual $RV_{t+1}$ (Realized)", color="#1f77b4", alpha=0.75, linewidth=1.2)
    ax1.plot(df["date"], df["pred_ann"], label="HAR-RV-X Forecast", color="#ff7f0e", linestyle="--", alpha=0.9, linewidth=1.2)
    ax1.set_title("1. Annualized Realized Volatility Forecast vs. Actual (%)", fontweight="bold")
    ax1.set_ylabel("Annualized Volatility (%)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Panel 2: Scatter Plot & Calibration Line
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    ax2.scatter(df["pred_ann"], df["actual_ann"], alpha=0.4, color="#2ca02c", edgecolors="none", s=25)
    
    # 45-degree 1:1 reference line
    min_val = min(df["pred_ann"].min(), df["actual_ann"].min())
    max_val = max(df["pred_ann"].max(), df["actual_ann"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label="Perfect Calibration (1:1)")
    
    ax2.set_title("2. Forecast Calibration (Actual vs. Predicted)", fontweight="bold")
    ax2.set_xlabel("Predicted Volatility (%)")
    ax2.set_ylabel("Actual Volatility (%)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Panel 3: Residual Distribution (Histogram & KDE)
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    ax3.hist(df["residuals_ann"], bins=40, density=True, color="#d62728", alpha=0.6, edgecolor="white")
    df["residuals_ann"].plot(kind="kde", ax=ax3, color="#8c564b", linewidth=2, label="Residual Density")
    ax3.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax3.set_title("3. Residual Error Distribution ($RV_{actual} - RV_{pred}$)", fontweight="bold")
    ax3.set_xlabel("Error in Annualized Volatility (%)")
    ax3.set_ylabel("Density")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Panel 4: Autocorrelation of Residuals (ACF)
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    plot_acf(df["residuals_ann"], ax=ax4, lags=30, alpha=0.05, color="#9467bd", vlines_kwargs={"linewidth": 1.5})
    ax4.set_title("4. Residual Autocorrelation (ACF Lags 1-30)", fontweight="bold")
    ax4.set_xlabel("Lag (Days)")
    ax4.set_ylabel("Autocorrelation")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Visual diagnostic figure successfully saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    from src.models.volatility_engine import train_har_model
    
    master_path = os.path.join(PROCESSED_DIR, "master_feature_matrix.csv")
    
    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path)
        _, df_results = train_har_model(df_master, start_date="2023-01-01", include_microstructure=True)
        evaluate_and_plot_volatility(df_results)
    else:
        print(f"❌ Could not find {master_path}. Run master_pipeline.py first.")