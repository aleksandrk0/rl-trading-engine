"""
Демо пайплайна на СИНТЕТИЧЕСКИХ данных.
Запускается на CPU, любой ОС, без ключей / MetaTrader5 / GPU / реальных данных.

Показывает сквозной поток: данные -> КАУЗАЛЬНЫЕ признаки -> модель -> сигнал -> equity.
Модель здесь необученная и игрушечная — это демонстрация инженерии пайплайна,
а НЕ торговая стратегия. Реальные архитектуры — в models/.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


def make_synthetic_ohlc(n: int = 3000) -> pd.DataFrame:
    """Геометрическое случайное блуждание + слабый momentum-сигнал."""
    eps = np.random.normal(0, 1.0, n)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.6 * drift[i - 1] + eps[i]      # слабая автокорреляция
    rets = 0.0008 * drift
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close})


def causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Все признаки ТРЕЙЛИНГОВЫЕ (только прошлое): без center=True и без shift(-)."""
    f = pd.DataFrame(index=df.index)
    f["ret1"] = df["close"].pct_change()
    f["sma_fast"] = df["close"].rolling(10).mean() / df["close"] - 1.0
    f["sma_slow"] = df["close"].rolling(50).mean() / df["close"] - 1.0
    f["mom"] = df["close"].pct_change(10)
    f["vol"] = f["ret1"].rolling(20).std()
    return f.dropna()


class TinyNet(nn.Module):
    """Маленький классификатор направления (демо)."""
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    df = make_synthetic_ohlc()
    feats = causal_features(df)

    # Метка: знак СЛЕДУЮЩЕГО возврата. shift(-1) применяется ТОЛЬКО к метке, не к фичам.
    next_ret = df["close"].pct_change().shift(-1).loc[feats.index]

    X = torch.tensor(feats.values, dtype=torch.float32)
    net = TinyNet(X.shape[1])
    with torch.no_grad():
        proba_up = torch.softmax(net(X), dim=1)[:, 1].numpy()

    signal = np.where(proba_up > 0.5, 1.0, -1.0)
    pnl = signal * next_ret.values
    equity = np.nancumsum(pnl)

    plt.figure(figsize=(9, 4))
    plt.plot(equity)
    plt.title("Demo equity — synthetic data, UNTRAINED model (пайплайн, не стратегия)")
    plt.xlabel("bar")
    plt.ylabel("cumulative return")
    plt.tight_layout()
    plt.savefig("demo_equity.png", dpi=110)

    print(f"OK: {len(feats)} баров, {X.shape[1]} каузальных признаков, "
          f"forward pass прошёл на устройстве '{next(net.parameters()).device}'.")
    print("График сохранён -> demo_equity.png")


if __name__ == "__main__":
    main()
