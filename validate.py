"""
Каркас валидации модели направления — против УТЕЧКИ и на РОБАСТНОСТЬ.

Две проверки, отвечающие на главный вопрос интервьюера про точность:
  1. SHUFFLE-LABEL: обучаем на СЛУЧАЙНО перемешанных метках. Если accuracy
     остаётся заметно выше 0.5 — в признаках есть утечка (модель "запоминает"
     через протёкшую информацию). Если падает к ~0.5 — утечки нет.
  2. WALK-FORWARD: оцениваем accuracy на нескольких последовательных окнах и
     смотрим РАСПРЕДЕЛЕНИЕ (медиана + разброс), а не одну цифру.

Замените load_data() на загрузку СВОИХ реальных X (n, d) и y (n,).
Зависимости — только ядро (numpy, torch). Запускается на CPU.
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)


def load_data():
    """ЗАГЛУШКА: синтетика со слабым реальным сигналом (нет утечки).
    Замените на свои признаки/метки.
    Возвращает X: (n, d) float32, y: (n,) int64 (0/1)."""
    n, d = 6000, 16
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype("float32")
    logit = 0.7 * X[:, 0] + 0.4 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype("int64")
    return X, y


class Probe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def _train_eval(Xtr, ytr, Xva, yva, epochs=15):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Probe(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xva_t).argmax(1).cpu().numpy()
    return float((pred == yva).mean())


def shuffle_label_test(X, y):
    """Если на перемешанных метках accuracy >> 0.5 — есть утечка."""
    cut = int(len(X) * 0.8)
    real = _train_eval(X[:cut], y[:cut], X[cut:], y[cut:])
    y_shuf = y.copy(); np.random.shuffle(y_shuf)
    leaked = _train_eval(X[:cut], y_shuf[:cut], X[cut:], y_shuf[cut:])
    print(f"[shuffle-label] real={real:.3f}  shuffled={leaked:.3f}")
    verdict = "OK: утечки не видно" if leaked < 0.55 else "ВНИМАНИЕ: возможна утечка"
    print(f"               -> {verdict} (на перемешанных метках ожидаем ~0.50)")


def walk_forward(X, y, n_folds=5):
    """Распределение accuracy по последовательным окнам (хронологически)."""
    accs = []
    fold = len(X) // (n_folds + 1)
    for k in range(1, n_folds + 1):
        tr_end = fold * k
        va_end = fold * (k + 1)
        accs.append(_train_eval(X[:tr_end], y[:tr_end], X[tr_end:va_end], y[tr_end:va_end]))
    accs = np.array(accs)
    print(f"[walk-forward] окна={np.round(accs, 3).tolist()}")
    print(f"               median={np.median(accs):.3f}  "
          f"min={accs.min():.3f}  max={accs.max():.3f}")
    print("               -> в README/резюме указывайте именно медиану+разброс, не одну цифру.")


if __name__ == "__main__":
    X, y = load_data()
    print(f"Данные: X={X.shape}, баланс классов={y.mean():.3f}")
    shuffle_label_test(X, y)
    walk_forward(X, y)
