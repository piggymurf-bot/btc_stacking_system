import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_ffd_weights(d: float, thres: float = 1e-3) -> np.ndarray:
  """Generates fixed-width window weights for fractional differentiation."""
  w = [1.0]
  k = 1
  while True:
    w_k = -w[-1] / k * (d - k + 1)
    if abs(w_k) < thres:
      break
    w.append(w_k)
    k += 1
  return np.array(w[::-1])  # Reverse for convolution alignment


def frac_diff_ffd(
    series: pd.Series, d: float, thres: float = 1e-3
) -> pd.Series:
  """Applies Fixed-Width Window Fractional Differentiation (FFD) to a series."""
  weights = get_ffd_weights(d, thres)
  width = len(weights)
  res = {}

  # Convolution over rolling window
  for i in range(width - 1, len(series)):
    window = series.iloc[i - width + 1 : i + 1]
    res[series.index[i]] = np.dot(weights, window)

  res_series = pd.Series(res)
  return res_series


def find_optimal_d(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.05,
    p_thres: float = 0.05,
) -> float:
  """Finds the minimum differentiation order d* that makes the series stationary (ADF p-value < p_thres)."""
  series_clean = series.dropna()

  for d in np.arange(0.0, max_d + step, step):
    if d == 0:
      ffd_series = series_clean
    else:
      ffd_series = frac_diff_ffd(series_clean, d=d).dropna()

    if len(ffd_series) < 30:
      continue

    # Run Augmented Dickey-Fuller stationarity test
    p_val = adfuller(ffd_series, autolag='AIC')[1]

    if p_val < p_thres:
      print(
          f'    [FracDiff] Optimal d* = {d:.2f} achieved stationarity (ADF'
          f' p-val = {p_val:.4f})'
      )
      return round(d, 2)

  return 1.0