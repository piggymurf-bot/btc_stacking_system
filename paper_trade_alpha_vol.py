import os
import sqlite3
import datetime
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
DB_PATH = "data/paper_trading_alpha_vol.db"
ALPHA_MODEL_PATH = "models/alpha_lgb.pkl"  # Adjust to your saved model path
VOLAT_MODEL_PATH = "models/volat_lgb.pkl"  # Adjust to your saved model path
MASTER_DATA_PATH = "data/processed/master_feature_addlhar_alpha.csv"
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 10000.0  # $10,000 Starting paper capital

# Strategy Parameters (Optimized & Stress-Tested)
QUANTILE_UPPER = 0.80
QUANTILE_LOWER = 0.19
STOP_LOSS_PCT = 0.08       # 8% Stop-Loss bracket
TAKE_PROFIT_PCT = 0.10     # 10% Take-Profit bracket
TRANSACTION_COST = 0.001   # 10 bps fee model
HOLDING_PERIOD_DAYS = 14

# -----------------------------------------------------------------------------
# 1. DATABASE MANAGEMENT (State & Position Tracking)
# -----------------------------------------------------------------------------
def init_db():
    """Initializes SQLite database to store portfolio balance, active trade brackets, and logs."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for portfolio & active trade state
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            cash REAL,
            btc_units REAL,
            last_close REAL,
            total_equity REAL,
            position REAL,
            entry_price REAL,
            entry_date TEXT,
            holding_days_elapsed INTEGER,
            stop_price REAL,
            target_price REAL,
            updated_at TEXT
        )
    """)

    # Table for historical trade execution logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            price REAL,
            units REAL,
            cost_usd REAL,
            target_position REAL,
            exit_reason TEXT
        )
    """)

    # Initialize portfolio if empty
    cursor.execute("SELECT COUNT(*) FROM portfolio")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            """INSERT INTO portfolio VALUES (1, ?, 0.0, 0.0, ?, 0.0, 0.0, '', 0, 0.0, 0.0, ?)""",
            (INITIAL_CAPITAL, INITIAL_CAPITAL, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()

    conn.close()


def get_portfolio_state():
    """Retrieves current paper portfolio and active trade state."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM portfolio WHERE id=1", conn)
    conn.close()
    return df.iloc[0].to_dict()


def update_portfolio_state(cash, btc_units, last_close, position, entry_price, entry_date, days_elapsed, stop_price, target_price):
    """Updates paper balance and trade lifecycle state."""
    total_equity = cash + (btc_units * last_close)
    updated_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE portfolio SET cash=?, btc_units=?, last_close=?, total_equity=?, position=?, 
           entry_price=?, entry_date=?, holding_days_elapsed=?, stop_price=?, target_price=?, updated_at=? WHERE id=1""",
        (cash, btc_units, last_close, total_equity, float(position), float(entry_price), 
         str(entry_date), int(days_elapsed), float(stop_price), float(target_price), updated_at)
    )
    conn.commit()
    conn.close()


def log_trade(action, price, units, cost_usd, target_pos, exit_reason=""):
    """Logs trade activity to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trade_logs (timestamp, action, price, units, cost_usd, target_position, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), action, price, units, cost_usd, target_pos, exit_reason)
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# 2. DATA INGESTION & ROLLING QUANTILE SIGNAL GENERATION
# -----------------------------------------------------------------------------
def generate_live_signal():
    """
    Loads master dataset, evaluates latest model predictions, and computes 
    strict point-in-time rolling quantiles to prevent look-ahead bias.
    """
    if not os.path.exists(MASTER_DATA_PATH):
        raise FileNotFoundError(f"Master feature file not found at {MASTER_DATA_PATH}.")

    master_df = pd.read_csv(MASTER_DATA_PATH, parse_dates=["date"])
    master_df = master_df.sort_values("date").reset_index(drop=True)

    # Load trained models (or load pre-generated OOS predictions dataframe if inference runs separately)
    # For live deployment, you can load your trained LightGBM models and run predict() on the latest feature row.
    # Here we assume you evaluate the latest row from your feature pipeline:
    
    # Placeholder: if you use saved out-of-sample predictions for live simulation testing:
    alpha_df = pd.read_csv("data/processed/alpha_wfo_out_of_sample_predictions.csv", parse_dates=["date"])
    volat_df = pd.read_csv("data/processed/volat_wfo_out_of_sample_predictions.csv", parse_dates=["date"])

    merged_df = pd.merge(
        alpha_df[["date", "pred_reg"]], 
        volat_df[["date", "pred_reg"]], 
        on="date", 
        suffixes=("_alpha", "_vol")
    ).sort_values("date").reset_index(drop=True)
   
    merged_df = merged_df.rename(columns={
        "pred_reg_alpha": "alpha_pred",
        "pred_reg_vol": "vol_pred"
    })

    merged_df = pd.merge(merged_df, master_df[["date", "open", "high", "low", "close"]], on="date", how="inner")
    merged_df["vol_pred"] = merged_df["vol_pred"].clip(lower=0.01)

    # Reconstruct strict expanding rolling quantiles up to the latest available timestamp
    all_alphas = merged_df["alpha_pred"].tolist()
    past_alphas = pd.Series(all_alphas[:-1])  # Exclude current live row to prevent lookahead
    
    alpha_upper = past_alphas.quantile(QUANTILE_UPPER)
    alpha_lower = past_alphas.quantile(QUANTILE_LOWER)

    latest_row = merged_df.iloc[-1]
    current_alpha = latest_row["alpha_pred"]
    current_vol = latest_row["vol_pred"]
    current_price = latest_row["close"]
    current_high = latest_row["high"]
    current_low = latest_row["low"]
    current_date = latest_row["date"]

    # Compute Risk-Parity Position Size: Pos = alpha / (vol^2)
    raw_pos = current_alpha / (current_vol ** 2)
    pos = 0.0
    if current_alpha > alpha_upper:
        pos = raw_pos
    elif current_alpha < alpha_lower:
        pos = raw_pos
        
    pos = float(np.clip(pos, -1.0, 1.0))

    return pos, current_price, current_high, current_low, current_date


# Safeguard against running multiple times on the same day
def has_traded_today():
    """Checks if an evaluation/trade action was already logged today (UTC)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM trade_logs WHERE DATE(timestamp) = ?", (today_str,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


# -----------------------------------------------------------------------------
# 3. DAILY PAPER EXECUTION & PATH-DEPENDENT SL/TP ENGINE
# -----------------------------------------------------------------------------
def execute_paper_trade():
    """
    Evaluates active trade path-dependent SL/TP brackets, checks for 14-day time expiration,
    or executes target portfolio rebalancing.
    """
    init_db()
    
    state = get_portfolio_state()
    cash = state["cash"]
    btc_units = state["btc_units"]
    active_pos = state["position"]
    entry_price = state["entry_price"]
    # Map state variables cleanly to avoid scoping errors
    days_elapsed = int(state["holding_days_elapsed"])
    stop_price = float(state["stop_price"])
    target_price = float(state["target_price"])
    entry_date = state["entry_date"]

    # Fetch latest market data and rolling signal
    target_pos, current_price, current_high, current_low, current_date = generate_live_signal()
    total_equity = cash + (btc_units * current_price)

    print(f"\n--- 🛡️ Alpha/Vol Paper Execution Engine ({current_date.strftime('%Y-%m-%d')}) ---")
    print(f"BTC Close Price   : ${current_price:,.2f} (High: ${current_high:,.2f} | Low: ${current_low:,.2f})")
    print(f"Target Position   : {target_pos * 100:.1f}%")
    print(f"Portfolio Equity  : ${total_equity:,.2f} (Cash: ${cash:,.2f}, BTC: {btc_units:.4f})")

    exit_triggered = False
    exit_reason = ""

    # -------------------------------------------------------------------------
    # ACTIVE TRADE MONITORING (Intra-Period Path-Dependent SL / TP & Expiration)
    # -------------------------------------------------------------------------
    if active_pos != 0.0:
        days_elapsed += 1
        print(f"Active Trade Duration: Day {days_elapsed} of {HOLDING_PERIOD_DAYS}")

        # Check Stop-Loss and Take-Profit triggers based on daily High/Low bars
        if active_pos > 0:  # LONG TRADE
            if current_low <= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                current_price = stop_price  # Fill at stop price
            elif current_high >= target_price:
                exit_triggered = True
                exit_reason = "take_profit"
                current_price = target_price  # Fill at target price
        else:  # SHORT TRADE
            if current_high >= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                current_price = stop_price
            elif current_low <= target_price:
                exit_triggered = True
                exit_reason = "take_profit"
                current_price = target_price

        # Check Time Expiration (14 days reached)
        if not exit_triggered and days_elapsed >= HOLDING_PERIOD_DAYS:
            exit_triggered = True
            exit_reason = "time_expiration"

        if exit_triggered:
            print(f"🚨 TRADE EXIT TRIGGERED: Reason -> {exit_reason.upper()} @ ${current_price:,.2f}")
            
            # Liquidate current position back to cash
            proceeds = (btc_units * current_price) * (1 - TRANSACTION_COST) if btc_units > 0 else 0
            new_cash = cash + proceeds if btc_units > 0 else cash - abs(btc_units) * current_price * (1 + TRANSACTION_COST)
            
            log_trade("EXIT", current_price, btc_units, abs(btc_units * current_price), 0.0, exit_reason)
            update_portfolio_state(new_cash, 0.0, current_price, 0.0, 0.0, "", 0, 0.0, 0.0)
            return

    # -------------------------------------------------------------------------
    # REBALANCING / NEW ENTRY LOGIC (14-Day Schedule or Flat State)
    # -------------------------------------------------------------------------
    if active_pos == 0.0 and target_pos != 0.0:
        # Open New Position
        target_usd = total_equity * abs(target_pos)
        units_to_trade = target_usd / current_price
        cost_with_fee = target_usd * (1 + TRANSACTION_COST)

        if cost_with_fee > cash:
            units_to_trade = cash / (current_price * (1 + TRANSACTION_COST))
            target_usd = cash

        new_cash = cash - (units_to_trade * current_price * (1 + TRANSACTION_COST))
        new_btc = units_to_trade if target_pos > 0 else -units_to_trade

        # Set SL and TP Brackets
        if target_pos > 0:
            new_stop = current_price * (1 - STOP_LOSS_PCT)
            new_target = current_price * (1 + TAKE_PROFIT_PCT)
        else:
            new_stop = current_price * (1 + STOP_LOSS_PCT)
            new_target = current_price * (1 - TAKE_PROFIT_PCT)

        update_portfolio_state(
            new_cash, new_btc, current_price, target_pos, 
            entry_price=current_price, entry_date=current_date.strftime('%Y-%m-%d'), 
            days_elapsed=1, stop_price=new_stop, target_price=new_target
        )
        log_trade("OPEN_LONG" if target_pos > 0 else "OPEN_SHORT", current_price, units_to_trade, target_usd, target_pos, "new_entry")
        print(f"🟢 OPENED POSITION: {target_pos*100:.1f}% allocation @ ${current_price:,.2f} | SL: ${new_stop:,.2f} | TP: ${new_target:,.2f}")

    elif active_pos != 0.0 and days_elapsed >= HOLDING_PERIOD_DAYS:
        # Rebalance at 14-day interval boundary
        print("⏳ 14-Day holding window expired. Re-evaluating rebalance target...")
        update_portfolio_state(total_equity, 0.0, current_price, 0.0, 0.0, "", 0, 0.0, 0.0)
    else:
        print("✅ Holding current active position. No rebalance action required today.")
        update_portfolio_state(cash, btc_units, current_price, active_pos, entry_price, entry_date, days_elapsed, stop_price, target_price)

# -----------------------------------------------------------------------------
# MAIN RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    execute_paper_trade()