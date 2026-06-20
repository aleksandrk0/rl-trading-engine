"""Генерирует крошечный синтетический OHLC-CSV для демо/тестов (без реальных данных)."""
import numpy as np
import pandas as pd
import pathlib

np.random.seed(42)
n = 1000
rets = 0.0008 * np.random.normal(0, 1, n).cumsum()
close = 100 * np.exp(rets)
ts = pd.date_range("2024-01-01", periods=n, freq="5min")
df = pd.DataFrame({
    "timestamp": ts,
    "open": np.r_[close[0], close[:-1]].round(5),
    "high": (close * 1.0005).round(5),
    "low": (close * 0.9995).round(5),
    "close": close.round(5),
    "volume": np.random.randint(50, 500, n),
})
out = pathlib.Path(__file__).with_name("ohlc_sample.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df)} bars -> {out}")
