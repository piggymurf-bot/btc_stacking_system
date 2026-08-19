import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.backtest.engine import VectorizedBacktester

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Alpha Engine | Walk-Forward Stacking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished metric cards
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2A2E39;
    }
    .stMetricLabel { font-size: 14px !important; color: #848E9C !important; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ Walk-Forward Stacking & FracDiff Strategy Engine")
st.caption(
    "Interactive quantitative framework featuring Fractional Differentiation"
    " memory preservation, Meta-Learner probability stacking, and dynamic ATR"
    " risk controls."
)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Strategy Controls & Preset Modes")

# Quick Preset Buttons
preset = st.sidebar.radio(
    "Select Target Risk Profile:",
    options=[
        "🛡️ Defensive (Capital Preservation)",
        "🚀 Aggressive (Upside Capture)",
        "⚙️ Custom Configuration",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameter Tuning")

# Set defaults based on preset selection
if "Defensive" in preset:
  default_min_prob = 0.55
  default_scale = 2.5
  default_rsi = 65.0
  default_atr = 0.8
elif "Aggressive" in preset:
  default_min_prob = 0.51
  default_scale = 3.5
  default_rsi = 72.0
  default_atr = 1.5
else:
  default_min_prob = 0.53
  default_scale = 3.0
  default_rsi = 70.0
  default_atr = 1.0

# Interactive Sliders
min_prob = st.sidebar.slider(
    "Minimum Signal Probability Floor",
    min_value=0.50,
    max_value=0.65,
    value=default_min_prob,
    step=0.01,
    help="Probability floor required before opening position.",
)

max_scale = st.sidebar.slider(
    "Position Scaling Multiplier",
    min_value=1.0,
    max_value=5.0,
    value=default_scale,
    step=0.1,
    help="Speed at which allocation scales to 1.0 based on model confidence.",
)

rsi_filter = st.sidebar.slider(
    "RSI Overbought Cutoff Gate",
    min_value=60.0,
    max_value=85.0,
    value=default_rsi,
    step=1.0,
    help="Forces target position to 0 when RSI exceeds this threshold.",
)

atr_mult = st.sidebar.slider(
    "ATR Trailing Stop Multiplier",
    min_value=0.5,
    max_value=3.0,
    value=default_atr,
    step=0.1,
    help="Distance of trailing stop loss in ATR multiples.",
)


# -----------------------------------------------------------------------------
# BACKTEST EXECUTION ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_run_backtest(
    min_p, scale, rsi_cap, atr_m, preds_path="data/processed/holdout_predictions.csv"
):
  if not os.path.exists(preds_path):
    return None, None

  tester = VectorizedBacktester(predictions_path=preds_path)
  df_res = tester.run_backtest(
      sizing_mode="dynamic",
      min_probability=min_p,
      max_position_scale=scale,
      rsi_max_filter=rsi_cap,
      atr_multiplier=atr_m,
      smooth_window=2,
      min_rebalance_delta=0.04,
  )
  metrics = tester.compute_performance_metrics()
  return df_res, metrics


df_results, metrics = load_and_run_backtest(
    min_prob, max_scale, rsi_filter, atr_mult
)

if df_results is None:
  st.error(
      "❌ Holdout predictions file not found! Please run your training pipeline"
      " first (`python main.py --train`)."
  )
  st.stop()

# -----------------------------------------------------------------------------
# SECTION 1: KEY PERFORMANCE INDICATORS (KPIs)
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Strategy Net Return",
    f"{metrics['Total Return Strategy (%)']}%",
    delta=f"{round(metrics['Total Return Strategy (%)'] - metrics['Total Return BTC Benchmark (%)'], 2)}% vs BTC",
)
col2.metric("BTC Benchmark", f"{metrics['Total Return BTC Benchmark (%)']}%")
col3.metric("Annualized Sharpe", f"{metrics['Annualized Sharpe Ratio']}")
col4.metric(
    "Max Drawdown",
    f"{metrics['Max Drawdown (%)']}%",
    delta_color="inverse",
)
col5.metric("Portfolio Turnover", f"{metrics['Total Portfolio Turnover (x)']}x")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 2: INTERACTIVE CHARTS
# -----------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📈 Dynamic Backtest UI", "🗺️ Parameter Plateau Detector"])

with tab1:
  st.subheader("📊 Interactive Equity Curve & Allocation Dynamics")

  fig = make_subplots(
      rows=2,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.05,
      row_heights=[0.75, 0.25],
      subplot_titles=("Cumulative Portfolio Value ($)", "Active Position Size (0.0 to 1.0)"),
  )

  # Plot Strategy Equity Curve
  fig.add_trace(
      go.Scatter(
          x=df_results.index,
          y=df_results["equity_strategy"],
          name="Dynamic Strategy",
          line=dict(color="#2962FF", width=2),
      ),
      row=1,
      col=1,
  )

  # Plot BTC Benchmark Equity Curve
  fig.add_trace(
      go.Scatter(
          x=df_results.index,
          y=df_results["equity_benchmark"],
          name="BTC Benchmark",
          line=dict(color="#787B86", width=1.5, dash="dash"),
      ),
      row=1,
      col=1,
  )

  # Plot Position Allocation Area
  fig.add_trace(
      go.Scatter(
          x=df_results.index,
          y=df_results["position"],
          name="Position Exposure",
          fill="tozeroy",
          line=dict(color="rgba(41, 98, 255, 0.5)", width=1),
      ),
      row=2,
      col=1,
  )

  fig.update_layout(
      template="plotly_dark",
      height=600,
      margin=dict(l=20, r=20, t=40, b=20),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
  )

  st.plotly_chart(fig, use_container_width=True)

with tab2:
  st.subheader("Interactive 2D Parameter Surface")
  
  col_x, col_y = st.columns(2)
  with col_x:
    param_x = st.selectbox(
        "X-Axis Parameter (Columns)",
        options=["atr_multiplier", "rsi_max_filter", "max_position_scale"],
        index=0  # Default: ATR
    )
  with col_y:
    param_y = st.selectbox(
        "Y-Axis Parameter (Rows)",
        options=["min_probability", "max_position_scale", "rsi_max_filter"],
        index=0  # Default: min_probability
    )

  if st.button("🚀 Run 2D Sensitivity Grid Sweep"):
    from src.backtest.sensitivity import SensitivityAnalyzer
    import plotly.express as px

    with st.spinner("Sweeping parameter combinations..."):
      analyzer = SensitivityAnalyzer()
      _, pivot_sharpe = analyzer.run_2d_grid(
          scale_range=np.linspace(1.5, 5.0, 8),
          atr_range=np.linspace(0.5, 2.5, 9),
          metric="Annualized Sharpe Ratio",
      )

      # Render Interactive Plotly Heatmap with clear labels
      fig_heatmap = px.imshow(
          pivot_sharpe,
          labels=dict(
              x="ATR Trailing Stop Multiplier (Stop Loss Distance)",
              y="Position Scaling Multiplier (Aggressiveness)",
              color="Sharpe Ratio",
          ),
          x=[f"{col:.2f}" for col in pivot_sharpe.columns],
          y=[f"{idx:.2f}" for idx in pivot_sharpe.index],
          color_continuous_scale="Viridis",
          aspect="auto",
          text_auto=".2f",  # Displays values inside cells
      )

      fig_heatmap.update_layout(
          title="<b>Annualized Sharpe Ratio Plateau</b>",
          template="plotly_dark",
          height=500,
          xaxis_title="<b>ATR Trailing Stop Multiplier</b>",
          yaxis_title="<b>Position Scaling Multiplier</b>",
      )

      # Force Y-axis orientation so 1.5 is at bottom, 5.0 is at top
      fig_heatmap.update_yaxes(autorange="reversed")

      st.plotly_chart(fig_heatmap, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 3: STRATEGY DIAGNOSTICS & DATA TABLE
# -----------------------------------------------------------------------------
with st.expander("🔍 View Raw Backtest Performance Logs"):
  st.dataframe(
      df_results[
          [
              "close",
              "y_pred_prob",
              "prob_smooth",
              "position",
              "strategy_net_return",
              "equity_strategy",
          ]
      ].tail(100)
  )