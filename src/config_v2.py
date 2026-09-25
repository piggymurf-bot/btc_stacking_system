FEATURE_GROUPS = {
    # -------------------------------------------------------------------------
    # Group 1: On-Chain Fundamentals & Supply Dynamics (Macro Volatility Drivers)
    # Source: BGeometrics
    # Base Model: Random Forest Regressor / LightGBM Regressor
    # -------------------------------------------------------------------------
    "onchain_features": [
        # 1. Fractional Differenced Non-Stationary Macro Metrics (Preserves Memory + Stationarity)
        "aviv_fracdiff",
        "mvrv_fracdiff",
        "nupl_fracdiff",
        "nrplbtc_log_fracdiff",
    
        # 2. Naturally Stationary/Mean-Reverting Level Series
        "sopr",             # Raw SOPR naturally mean-reverts around 1.0
        
        # 3. UTXO Realized Cap Age Distribution (in % Shares)
        "age_0d_1d_share", "age_1d_1w_share", "age_1w_1m_share", "age_1m_3m_share", 
        "age_3m_6m_share", "age_6m_1y_share", "age_1y_2y_share", "age_2y_3y_share", 
        "age_3y_4y_share", "age_4y_5y_share", "age_5y_7y_share", "age_7y_10y_share", 
        "age_10y_share",
        
        # 4. Short-Term Acceleration / Velocity Signals (1-Day Changes)
        "mvrv_daily_diff",
        "nupl_daily_diff",
        "sopr_daily_diff",
        "age_0d_1d_share_daily_diff",
        "age_1d_1w_share_daily_diff",
        "age_1w_1m_share_daily_diff",
        "age_1m_3m_share_daily_diff",
        "age_3m_6m_share_daily_diff",
        "age_6m_1y_share_daily_diff",
        "age_1y_2y_share_daily_diff",
        "age_2y_3y_share_daily_diff",
        "age_3y_4y_share_daily_diff",
        "age_4y_5y_share_daily_diff",
        "age_5y_7y_share_daily_diff",
        "age_7y_10y_share_daily_diff",
        "age_10y_share_daily_diff",
    ],
    
    # -------------------------------------------------------------------------
    # Group 2: Derivatives, Positioning & Microstructure (Squeeze & Order Book Volatility)
    # Source: Binance Vision Futures Metrics + Funding + Book Depth
    # Base Model: Ridge / ElasticNet / LightGBM
    # -------------------------------------------------------------------------
    "derivatives_microstructure_features": [
        # Open Interest & Positioning
        #"open_interest_usd",
        "oi_pct_change_1d",
        "oi_pct_change_7d",
        "long_short_ratio",
        "ls_ratio_zscore_14d",
        "taker_buy_sell_ratio",
        
        # Funding Dynamics
        "funding_rate_daily_sum",
        "funding_rate_daily_mean",
        "funding_rate_daily_max",
        
        # Order Book Microstructure (L2 Depth & Imbalance)
        "obi_mean_daily",
        "obi_std_daily",
        #"book_depth_1m_mean",
        "depth_ratio_30d",
        "bid_ask_depth_ratio",
    ],
    
    # -------------------------------------------------------------------------
    # Group 3: Realized, Estimator & Implied Volatility (Short-Term Volatility Dynamics)
    # Source: Binance Vision 1m Klines + BGeometrics Technicals + Deribit DVOL
    # Base Model: LightGBM Regressor / LHAR-RV
    # -------------------------------------------------------------------------
    "technicals_and_volatility_features": [
        # ---------------------------------------------------------------------
        # 1. Intraday High-Frequency Realized Volatility (Binance 1m Klines)
        # ---------------------------------------------------------------------
        "rv_1m_daily",            # Pure 1m intraday Realized Volatility (1-day)
        "rv_w",                   # HAR-RV: 7-day rolling mean of RV
        "rv_m",                   # HAR-RV: 30-day rolling mean of RV
        "rv_pos_1m",              # Upside Semivariance
        "rv_neg_1m",              # Downside Semivariance
        #"rv_asymmetry_ratio",    # Log(rv_pos / rv_neg)
        "rv_asymmetry_ratio_d",   # Daily RV asymmetry
        "rv_asymmetry_ratio_w",   # Weekly RV asymmetry
    
        # ---------------------------------------------------------------------
        # 2. Range-Based Volatility Estimators
        # ---------------------------------------------------------------------
        "parkinson_vol",          # High-Low Range Volatility
        "garman_klass_vol",       # Open-High-Low-Close Volatility
        "hl_spread_pct",          # (High - Low) / Close
        "atr_14_pct",             # ATR(14) / Close
    
        # ---------------------------------------------------------------------
        # 3. Deribit Implied Volatility (DVOL Dynamics & Risk Premium)
        # ---------------------------------------------------------------------
        "dvol_close",             # Raw DVOL Index
        "dvol_daily_diff",        # 1-day DVOL difference
        "iv_rv_spread",           # Volatility Risk Premium: (dvol_close / 100) - rv_1m_daily
        "dvol_rv_ratio",          # Volatility Ratio: (dvol_close / 100) / (rv_1m_daily + 1e-8)
        "dvol_zscore_14d",        # 14-day DVOL Z-Score
    
        # ---------------------------------------------------------------------
        # 4. Technical Momentum & Trend Indicators (Binance Daily OHLCV)
        # ---------------------------------------------------------------------
        "close_fracdiff",         # Fractionally Differentiated Stationarized Price
        "close_log_return_1d",    # 1-day Log Return
        "close_log_return_7d",    # 7-day Log Return
        "vwap_dist_pct",          # (Close - VWAP) / VWAP
        "taker_buy_ratio",        # Aggressor Buy Volume Ratio
        "volume_ratio_20d",       # Volume / SMA_20(Volume)
        "rsi_14",                 # Relative Strength Index (14)
        "macd_hist_pct",              # MACD Histogram (12, 26, 9)
        "bollinger_pct_b",        # (Close - Lower_Band) / (Upper_Band - Lower_Band)
        "dist_from_sma20",        # (Close - SMA_20) / SMA_20
        "dist_from_sma200",       # (Close - SMA_200) / SMA_200
        "sma7_to_sma50_ratio"     # SMA_7 / SMA_50
    ],
}