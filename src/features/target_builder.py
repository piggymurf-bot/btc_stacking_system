import pandas as pd


def add_target_variable(
    df: pd.DataFrame,
    close_col: str = 'Close',
    sma_window: int = 7,
    horizon_days: int = 3,
) -> pd.DataFrame:
  """Calculates 3-day forward SMA movement to generate binary target y.

  Target: 1.0 if SMA7 in `horizon_days` > SMA7 today, else 0.0.
  Trims the last `horizon_days` rows to remove forward NaN values.
  """
  data = df.copy()

  if close_col not in data.columns:
    raise KeyError(
        f"Column '{close_col}' not found. Ensure raw close price is retained in"
        ' yfinance features.'
    )

  # 1. Compute rolling SMA7 on historical close price
  sma = data[close_col].rolling(window=sma_window).mean()

  # 2. Shift SMA BACKWARDS by horizon_days to look into the future
  future_sma = sma.shift(-horizon_days)

  # 3. Create binary classification target
  data['target'] = (future_sma > sma).astype(float)

  # 4. Drop the last `horizon_days` rows where future_sma is NaN
  data = data.iloc[:-horizon_days].copy()

  return data