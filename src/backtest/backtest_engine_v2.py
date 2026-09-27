import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.config_v2 import FEATURE_GROUPS

ONCHAIN_FEATURES = FEATURE_GROUPS["onchain_features"]
DERIVATIVES_FEATURES = FEATURE_GROUPS["derivatives_microstructure_features"]
TECHNICALS_VOL_FEATURES = FEATURE_GROUPS["technicals_and_volatility_features"]

def compute_indicators(df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    prev_close = df["close"].shift(1)
    if "high" in df.columns and "low" in df.columns:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        df["true_range"] = (df["close"] - prev_close).abs()

    df["atr_14"] = df["true_range"].rolling(window=atr_period, min_periods=1).mean()
    return df

def run_pipeline(
    data_path: str = "data/processed/master_feature_addlhar_alpha.csv",
    output_path: str = "data/processed/final_rigorous_hybrid_backtest.csv",
    alpha_target: str = "target_fwd_cum_ret_1d",
    vol_target: str = "target_rv",
    alpha_multiplier: float = 3.5,
    upside_vol_damping_factor: float = 0.50,
    rsi_max_filter: float = 75.0,
    atr_multiplier: float = 1.0,
    min_rebalance_delta: float = 0.05,  # Prevents excessive micro-turnover
    transaction_cost_bps: float = 10.0
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}.")

    print(f"📥 Loading master dataset from: {data_path}")
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError("Dataset must contain a 'close' column.")

    vol_features = list(set(DERIVATIVES_FEATURES + TECHNICALS_VOL_FEATURES))
    required_cols = [alpha_target, vol_target, "date", "close"] + ONCHAIN_FEATURES + vol_features
    model_df = df.dropna(subset=required_cols).reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=5)
    oof_alpha = np.full(len(model_df), np.nan)
    oof_vol = np.full(len(model_df), np.nan)

    print("\n🔄 Executing Out-of-Fold Training for Dual Engines...")

    # 1. On-Chain Alpha Engine
    X_alpha = model_df[ONCHAIN_FEATURES]
    y_alpha = model_df[alpha_target]
    alpha_params = {"objective": "regression", "metric": "rmse", "n_estimators": 500, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1}
    
    for train_idx, val_idx in tscv.split(X_alpha):
        model = lgb.LGBMRegressor(**alpha_params)
        model.fit(X_alpha.iloc[train_idx], y_alpha.iloc[train_idx], 
                  #eval_set=[(X_alpha.iloc[val_idx], y_alpha.iloc[val_idx])],
                  eval_X = X_alpha.iloc[val_idx],
                  eval_y = y_alpha.iloc[val_idx],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof_alpha[val_idx] = model.predict(X_alpha.iloc[val_idx])

    # 2. Volatility Engine
    X_vol = model_df[vol_features]
    y_vol = model_df[vol_target]
    vol_params = {"objective": "regression", "metric": "rmse", "n_estimators": 600, "learning_rate": 0.02, "max_depth": 4, "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1}
    
    for train_idx, val_idx in tscv.split(X_vol):
        model = lgb.LGBMRegressor(**vol_params)
        model.fit(X_vol.iloc[train_idx], y_vol.iloc[train_idx],
                  #eval_set=[(X_vol.iloc[val_idx], y_vol.iloc[val_idx])],
                  eval_X = X_vol.iloc[val_idx],
                  eval_y = y_vol.iloc[val_idx],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof_vol[val_idx] = model.predict(X_vol.iloc[val_idx])

    model_df["oof_alpha_pred"] = oof_alpha
    model_df["oof_vol_pred"] = oof_vol
    backtest_df = model_df.dropna(subset=["oof_alpha_pred", "oof_vol_pred"]).copy()
    backtest_df = compute_indicators(backtest_df)

    print("\n⚖️ Applying Rigorous Non-Lookahead Execution Loop...")
    
    backtest_df["safe_vol"] = backtest_df["oof_vol_pred"].clip(lower=0.005)
    effective_vol = backtest_df["safe_vol"].copy()
    upside_mask = backtest_df["oof_alpha_pred"] > 0
    effective_vol[upside_mask] = effective_vol[upside_mask] * upside_vol_damping_factor

    raw_weights = (backtest_df["oof_alpha_pred"] * alpha_multiplier) / effective_vol
    raw_weights = raw_weights.clip(lower=0.0, upper=1.0)

    vol_threshold = backtest_df["safe_vol"].quantile(0.90)
    raw_weights[backtest_df["safe_vol"] > vol_threshold] = 0.0
    raw_weights[backtest_df["rsi_14"] > rsi_max_filter] = 0.0
    backtest_df["raw_target_size"] = raw_weights

    # Stateful Execution Simulation Loop (Strictly No Look-Ahead)
    n = len(backtest_df)
    executed_positions = np.zeros(n)
    strategy_net_returns = np.zeros(n)
    turnover_history = np.zeros(n)

    prices = backtest_df["close"].values
    atrs = backtest_df["atr_14"].values
    targets = backtest_df["raw_target_size"].values
    cost_rate = transaction_cost_bps / 10000.0

    in_position = False
    active_size = 0.0
    highest_price = 0.0

    for i in range(1, n):
        prev_target = targets[i - 1]  # Yesterday's signal (no look-ahead)
        curr_price = prices[i]
        prev_price = prices[i - 1]
        curr_atr = atrs[i - 1]

        if in_position:
            # 1. Evaluate ATR Stop using YESTERDAY's highest price and ATR (No peeking)
            stop_price = highest_price - (atr_multiplier * curr_atr)

            if curr_price <= stop_price:
                # Event A: ATR Stop Triggered
                in_position = False
                exit_turnover = active_size
                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = exit_turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = exit_turnover
                active_size = 0.0
                executed_positions[i] = 0.0

            elif prev_target == 0.0:
                # Event B: Model signal or RSI gate forced exit
                in_position = False
                exit_turnover = active_size
                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = exit_turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = exit_turnover
                active_size = 0.0
                executed_positions[i] = 0.0

            else:
                # Event C: Rebalance with Deadband Check
                desired_size = prev_target
                size_change = abs(desired_size - active_size)

                if size_change >= min_rebalance_delta:
                    turnover = size_change
                    active_size = desired_size
                else:
                    turnover = 0.0

                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = turnover
                executed_positions[i] = active_size

                # Update highest price AFTER surviving today's close
                highest_price = max(highest_price, curr_price)

        else:
            # Event D: New Entry from Cash
            if prev_target > 0.0:
                in_position = True
                active_size = prev_target
                highest_price = curr_price  # Set initial peak on entry close

                entry_turnover = active_size
                raw_return = active_size * ((curr_price - prev_price) / prev_price)
                cost = entry_turnover * cost_rate

                strategy_net_returns[i] = raw_return - cost
                turnover_history[i] = entry_turnover
                executed_positions[i] = active_size

    backtest_df["position_weight"] = executed_positions
    backtest_df["turnover"] = turnover_history
    backtest_df["net_strategy_return"] = strategy_net_returns
    backtest_df["benchmark_return"] = backtest_df["close"].pct_change().fillna(0.0)

    # Compute Final Performance Metrics
    clean_rets = backtest_df["net_strategy_return"].dropna()
    wealth = (1 + clean_rets).cumprod()
    running_max = wealth.cummax()
    max_dd = ((wealth - running_max) / running_max).min()
    ann_return = clean_rets.mean() * 365
    ann_vol = clean_rets.std() * np.sqrt(365)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    print("\n" + "="*60)
    print("🚀 RIGOROUS HYBRID PIPELINE: FINAL PERFORMANCE")
    print("="*60)
    print(f"Cumulative Return (%)        : {(wealth.iloc[-1] - 1) * 100:.2f}%")
    print(f"Annualized Return (%)        : {ann_return * 100:.2f}%")
    print(f"Annualized Volatility (%)    : {ann_vol * 100:.2f}%")
    print(f"Sharpe Ratio                 : {sharpe:.2f}")
    print(f"Max Drawdown (%)             : {max_dd * 100:.2f}%")
    print(f"Average Daily Position Size  : {backtest_df['position_weight'].abs().mean():.4f}")
    print(f"Average Daily Turnover       : {backtest_df['turnover'].mean():.4f}")
    print("="*60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    backtest_df[["date", "position_weight", "net_strategy_return", "benchmark_return", "turnover", "rsi_14"]].to_csv(output_path, index=False)
    print(f"💾 Rigorous hybrid trade log saved to: {output_path}")

    return backtest_df

if __name__ == "__main__":
    run_pipeline()