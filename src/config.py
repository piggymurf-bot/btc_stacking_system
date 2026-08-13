FEATURE_GROUPS = {
    # -------------------------------------------------------------------------
    # Group 1: On-Chain Fundamentals & Supply Dynamics (32 features)
    # Target Model: XGBoost or Random Forest (Specialist in structural/macro regime shifts)
    # -------------------------------------------------------------------------
    "onchain_features": [
        "aviv",
        "mvrv",
        "nupl",
        "sopr",
        "age_0d_1d",
        "age_1d_1w",
        "age_1w_1m",
        "age_1m_3m",
        "age_3m_6m",
        "age_6m_1y",
        "age_1y_2y",
        "age_2y_3y",
        "age_3y_4y",
        "age_4y_5y",
        "age_5y_7y",
        "age_7y_10y",
        "age_10y",
        "mvrv_daily_diff",
        "nupl_daily_diff",
        "age_0d_1d_daily_diff",
        "age_1d_1w_daily_diff",
        "age_1w_1m_daily_diff",
        "age_1m_3m_daily_diff",
        "age_3m_6m_daily_diff",
        "age_6m_1y_daily_diff",
        "age_1y_2y_daily_diff",
        "age_2y_3y_daily_diff",
        "age_3y_4y_daily_diff",
        "age_4y_5y_daily_diff",
        "age_5y_7y_daily_diff",
        "age_7y_10y_daily_diff",
        "age_10y_daily_diff",
        #"macd_pct", 
        #"macdsignal_pct", 
        #"macdhist_pct", 
        #"nrplbtc_log"
    ],
    # -------------------------------------------------------------------------
    # Group 2: Derivatives & Market Microstructure (5 features)
    # Target Model: Logistic Regression or ElasticNet (Specialist in mean-reversion & position squeeze)
    # -------------------------------------------------------------------------
    "derivatives_features": [
        "oi_pct_change_1d",
        "ls_ratio_zscore_14d",
        "funding_rate_daily_sum",
        "funding_rate_daily_mean",
        "funding_rate_daily_max",
    ],
    # -------------------------------------------------------------------------
    # Group 3: Spot Price Momentum & Volatility Technicals (9 features)
    # Target Model: Linear SVM or Ridge Classifier (Specialist in trend-following)
    # -------------------------------------------------------------------------
    "technicals_features": [
        "yf_hl_spread_pct",
        "yf_atr14_pct",
        "yf_bollinger_pct_b",
        "yf_sma7_to_sma50_ratio",
        "yf_dist_from_sma20",
        "yf_rsi14",
        "yf_volume_ratio_20d",
        "yf_return_1d",
        "yf_return_7d",
    ],
}