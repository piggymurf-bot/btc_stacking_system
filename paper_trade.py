import os
import sqlite3
import datetime
import pandas as pd
import numpy as np
import ccxt
import joblib

from dotenv import load_dotenv

# Import custom features 
# DataLoader
from src.data.bgeometrics_loader import fetch_and_merge_bgeometrics
from src.data.binance_vision import fetch_and_process_binance_vision
from src.data.ccxt_funding import fetch_and_process_funding_rates
from src.data.yfinance_dprice import fetch_yfinance_data

# Featuring Dataset
from src.features.bgeometrics_extract import extract_bgeometrics_features
from src.features.binancevision_extract import extract_binancevision_features
from src.features.yfinance_extract import extract_yfinance_features
from src.features.fracdiff import frac_diff_ffd

# Custom class for the meta-learner
from src.models.base_pipelines import build_base_pipelines
from src.models.meta_learner import DomainStackingClassifier

from datetime import datetime, timezone

# Replace utcnow() with timezone-aware utc:
updated_at = datetime.now(timezone.utc).isoformat()

# Load API key for BGeometrics data
load_dotenv('token.env')
API_TOKEN = os.getenv('MY_TOKEN')

if not API_TOKEN:
  raise ValueError(
      "MY_TOKEN environment variable is missing or empty! Check local token.env"
      " or GitHub Secrets."
  )

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
DB_PATH = "data/paper_trading.db"
MODEL_PATH = "models/stacking_ensemble.pkl"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1d"
INITIAL_CAPITAL = 10000.0  # $10,000 Starting paper capital

# Strategy Parameters (Aggressive Preset)
#MIN_PROBABILITY = 0.53
#MAX_POSITION_SCALE = 3.5
#RSI_MAX_FILTER = 72.0
#ATR_MULTIPLIER = 1.25

# Strategy Parameters (Stabilized Preset)
MIN_PROBABILITY = 0.55
MAX_POSITION_SCALE = 3.0
RSI_MAX_FILTER = 75.0
ATR_MULTIPLIER = 1.0



# Directories for files
raw_dir: str = 'data/raw' 
out_dir: str = 'data/processed'

# -----------------------------------------------------------------------------
# 1. DATABASE MANAGEMENT (State Tracking)
# -----------------------------------------------------------------------------
def init_db():
    """Initializes SQLite database to store portfolio balance and trade logs."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for portfolio state
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            cash REAL,
            btc_units REAL,
            last_close REAL,
            total_equity REAL,
            updated_at TEXT
        )
    """)

    # Table for trade execution logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            price REAL,
            units REAL,
            cost_usd REAL,
            target_position REAL,
            model_prob REAL
        )
    """)

    # Initialize portfolio if empty
    cursor.execute("SELECT COUNT(*) FROM portfolio")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO portfolio VALUES (1, ?, 0.0, 0.0, ?, ?)",
            (INITIAL_CAPITAL, INITIAL_CAPITAL, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()

    conn.close()


def get_portfolio_state():
    """Retrieves current paper portfolio state."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM portfolio WHERE id=1", conn)
    conn.close()
    return df.iloc[0].to_dict()


def update_portfolio_state(cash, btc_units, last_close):
    """Updates paper balance after trade execution."""
    total_equity = cash + (btc_units * last_close)
    updated_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE portfolio SET cash=?, btc_units=?, last_close=?, total_equity=?, updated_at=? WHERE id=1",
        (cash, btc_units, last_close, total_equity, updated_at)
    )
    conn.commit()
    conn.close()


def log_trade(action, price, units, cost_usd, target_pos, model_prob):
    """Logs trade activity to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trade_logs (timestamp, action, price, units, cost_usd, target_position, model_prob) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), action, price, units, cost_usd, target_pos, model_prob)
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# 2. LIVE DATA INGESTION (CCXT)
# -----------------------------------------------------------------------------
def fetch_daily_data(symbol=SYMBOL, timeframe=TIMEFRAME, limit=300):
    """Fetches data for evaluate daily position"""
    print("📡 Fetching Data...")
    
    bg_df = fetch_and_merge_bgeometrics(API_TOKEN)

    # Extract date range from BGeometrics to synchronize all other data sources
    start_date = pd.to_datetime(bg_df['date'].iloc[0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(bg_df['date'].iloc[-1]).strftime('%Y-%m-%d')
    
    # Subtract 100 days so 50-day SMAs are fully calculated by 'start_date'
    yf_buffered_start = (pd.to_datetime(bg_df['date'].iloc[0]) - pd.Timedelta(days=100)).strftime(
      '%Y-%m-%d'
      )

    yf_buffered_end = (
            pd.to_datetime(bg_df['date'].iloc[-1]) + pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')

    fetch_and_process_binance_vision(symbol, start_date_str=start_date)

    fetch_and_process_funding_rates(start_date_str=start_date)

    fetch_yfinance_data(
        symbol='BTC-USD', start_date_str=yf_buffered_start, end_date_str=yf_buffered_end
    )
    
    
    file_path = os.path.join(raw_dir, 'bgeometrics_merged.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
    
    bg_df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    bg_df_processed = extract_bgeometrics_features(bg_df)
      
    file_path = os.path.join(raw_dir, 'binance_vision_metrics.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
    bnv_df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    bnv_df_processed = extract_binancevision_features(bnv_df)

    # Processing Funding Rate data
    file_path = os.path.join(raw_dir, 'daily_funding.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )    
    cctx_df_processed = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        
    # Processing Yahoo Finance data
    file_path = os.path.join(raw_dir, 'yfinance_data.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing raw data file: {file_path}. Run --download-data first.'
      )
      
    yf_df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    yf_df_processed = extract_yfinance_features(yf_df)
    
    daily_df = bg_df_processed.copy()

    daily_df = pd.merge(daily_df, bnv_df_processed, on='date', how='left')
    daily_df = pd.merge(daily_df, cctx_df_processed, on='date', how='left')
    daily_df = pd.merge(daily_df, yf_df_processed, on='date', how='left')
    
    daily_df = daily_df.ffill()
    
    daily_df = daily_df.dropna()
    
    non_stationary_cols = ['Close', 'mvrv', 'aviv', 'nrplbtc_log']
    
    # Get the optimal d from the saved file
    file_path = os.path.join(out_dir, 'optimal_d.csv')
    if not os.path.exists(file_path):
      raise FileNotFoundError(
          f'Missing file: {file_path}. Never running full pipeline? Run --all first'
      )
    d_opt_df = pd.read_csv(file_path)

    for col in non_stationary_cols:
        
        series_to_diff = (
            np.log(daily_df[col]) if col == 'Close' or (daily_df[col] > 0).all() else daily_df[col]
            )

        optimal_d = d_opt_df.loc[0,f'{col}']
       
        daily_df[f'{col}_fracdiff'] = frac_diff_ffd(series_to_diff, d=optimal_d)
    
    daily_df = daily_df.dropna()

    print("daily_data")
    print(daily_df.tail())

    return daily_df

# -----------------------------------------------------------------------------
# 3. LIVE SIGNAL GENERATION ENGINE
# -----------------------------------------------------------------------------
def generate_live_signal(df_market):
    """Processes daily data through the Meta-Learner model."""
    
    # Extract latest features
    latest_features = df_market.iloc[[-1]]

    # Load trained model weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train your model first.")
    
    model = joblib.load(MODEL_PATH)
    
    print("⚙️ Feeding data to the meta-learner...")

    # Predict probability for today's close
    prob = float(model.predict_proba_stack(latest_features)[:, 1][0])

    # Compute target position size using strategy rules
    if prob < MIN_PROBABILITY:
        target_pos = 0.0
    else:
        target_pos = float(np.clip(MAX_POSITION_SCALE * (prob - MIN_PROBABILITY), 0.0, 1.0))

    latest_close = float(df_market["Close"].iloc[-1])
    return prob, target_pos, latest_close

# Safeguard from run paper-trade several times a day

def has_traded_today():
    """Checks if a trade was already logged today (UTC)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    cursor.execute(
      "SELECT COUNT(*) FROM trade_logs WHERE DATE(timestamp) = ?", (today_str,)
      )
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


# -----------------------------------------------------------------------------
# 4. PAPER EXECUTION ENGINE
# -----------------------------------------------------------------------------
def execute_paper_trade():
    """Calculates rebalance requirements and executes simulated trade."""
    
    if has_traded_today():
      print("⏸️ Already executed a paper trade today. Skipping duplicate run.")
      return
    
    # 1. Get live price and prediction
    df_market = fetch_daily_data()
    model_prob, target_pos, current_price = generate_live_signal(df_market)
    
    init_db()
    state = get_portfolio_state()

    current_cash = state["cash"]
    current_btc = state["btc_units"]
    total_equity = current_cash + (current_btc * current_price)

    target_btc_usd = total_equity * target_pos
    current_btc_usd = current_btc * current_price
    delta_usd = target_btc_usd - current_btc_usd

    print("\n--- 📊 Live Paper Trade Execution ---")
    print(f"Current BTC Price : ${current_price:,.2f}")
    print(f"Model Signal Prob : {model_prob:.4f}")
    print(f"Target Position   : {target_pos * 100:.1f}% (${target_btc_usd:,.2f})")
    print(f"Current Portfolio : Cash=${current_cash:,.2f} | BTC={current_btc:.4f} (${current_btc_usd:,.2f})")
    print(f"Total Equity      : ${total_equity:,.2f}")

    # Minimum trade threshold to avoid tiny micro-rebalances ($20 min trade)
    MIN_TRADE_USD = 20.0

    if abs(delta_usd) < MIN_TRADE_USD:
        print("✅ Portfolio already aligned with target position. No rebalance needed.")
        update_portfolio_state(current_cash, current_btc, current_price)
        return

    # Execute BUY
    if delta_usd > 0:
        units_to_buy = delta_usd / current_price
        cost_with_fee = delta_usd * 1.0008  # Account for 0.08% taker fee + slippage
        
        if cost_with_fee > current_cash:
            units_to_buy = current_cash / (current_price * 1.0008)
            delta_usd = current_cash

        new_cash = current_cash - (units_to_buy * current_price * 1.0008)
        new_btc = current_btc + units_to_buy
        
        update_portfolio_state(new_cash, new_btc, current_price)
        log_trade("BUY", current_price, units_to_buy, delta_usd, target_pos, model_prob)
        print(f"🟢 EXECUTED BUY: {units_to_buy:.6f} BTC @ ${current_price:,.2f} (Value: ${delta_usd:,.2f})")

    # Execute SELL
    elif delta_usd < 0:
        usd_to_sell = abs(delta_usd)
        units_to_sell = usd_to_sell / current_price
        
        if units_to_sell > current_btc:
            units_to_sell = current_btc

        proceeds_after_fee = (units_to_sell * current_price) * (1 - 0.0008)
        new_cash = current_cash + proceeds_after_fee
        new_btc = current_btc - units_to_sell

        update_portfolio_state(new_cash, new_btc, current_price)
        log_trade("SELL", current_price, units_to_sell, usd_to_sell, target_pos, model_prob)
        print(f"🔴 EXECUTED SELL: {units_to_sell:.6f} BTC @ ${current_price:,.2f} (Value: ${usd_to_sell:,.2f})")


# -----------------------------------------------------------------------------
# MAIN RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    execute_paper_trade()