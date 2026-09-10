from functools import reduce
import os
import pandas as pd
import requests

URLS = [
    'https://api.bgeometrics.com/v1/aviv',
    'https://api.bgeometrics.com/v1/mvrv',
    'https://api.bgeometrics.com/v1/nupl',
    'https://api.bgeometrics.com/v1/sopr',
    'https://api.bgeometrics.com/v1/nrpl-btc',
    'https://api.bgeometrics.com/v1/hodl-waves-supply',
    'https://api.bgeometrics.com/v1/technical-indicators',
]

CSV_NAMES = [
    'aviv.csv',
    'mvrv.csv',
    'nupl.csv',
    'sopr.csv',
    'nrpl-btc.csv',
    'hodl-waves-supply.csv',
    'technical-indicators.csv',
]


def json_url_to_csv(
    url: str, output_csv_path: str, headers: dict = None
) -> pd.DataFrame:
  """Fetches JSON data from a URL, normalizes it, and saves it locally."""
  response = requests.get(url, headers=headers, timeout=15)
  response.raise_for_status()

  json_data = response.json()
  df = (
      pd.json_normalize(json_data)
      if isinstance(json_data, dict)
      else pd.DataFrame(json_data)
  )
  df.to_csv(output_csv_path, index=False)
  return df


def fetch_and_merge_bgeometrics(
    api_token: str, raw_dir: str = 'data/raw'
) -> pd.DataFrame:
  """Downloads all BGeometrics endpoints, cleans date columns, and merges on date."""
  os.makedirs(raw_dir, exist_ok=True)
  dfs = []

  for base_url, name in zip(URLS, CSV_NAMES):
    query_url = f'{base_url}?token={api_token}'
    out_path = os.path.join(raw_dir, name)

    try:
      df = json_url_to_csv(query_url, output_csv_path=out_path)

      # Standardize column names
      df.columns = df.columns.astype(str).str.lower()
      if 'unixts' in df.columns:
        df = df.drop(columns=['unixts'])

      # Clean and insert date column
      if 'd' in df.columns:
        df['date'] = pd.to_datetime(df['d'])
        df = df.drop(columns=['d'])
        df.insert(0, 'date', df.pop('date'))

      dfs.append(df)
    except Exception as e:
      print(f'Error fetching {name}: {e}')

  if not dfs:
    raise ValueError('No valid dataframes retrieved from BGeometrics.')

  # Merge all dataframes on date
  merged_matrix = reduce(
      lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs
  )
  merged_matrix = merged_matrix.sort_values(by='date').reset_index(drop=True)

  merged_path = os.path.join(raw_dir, 'bgeometrics_merged.csv')
  merged_matrix.to_csv(merged_path, index=False)
  print(
      f'Success! BGeometrics merged dataset saved to {merged_path} '
      f'({merged_matrix.shape[0]} rows x {merged_matrix.shape[1]} cols)'
  )

  return merged_matrix