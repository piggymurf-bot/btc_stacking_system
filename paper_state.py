import sqlite3
import pandas as pd

conn = sqlite3.connect("data/paper_trading.db")

print("--- Current Portfolio State ---")
print(pd.read_sql("SELECT * FROM portfolio", conn))

print("\n--- Execution History ---")
print(pd.read_sql("SELECT * FROM trade_logs", conn))

conn.close()