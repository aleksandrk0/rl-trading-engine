"""
Smoke-тест: доказывает, что репозиторий клонируется и работает на чистой машине
(CPU, любая ОС, без GPU/MetaTrader5/ключей/данных).
"""
import importlib
import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_torch_forward_on_cpu():
    """Минимальный forward pass — окружение torch исправно."""
    net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
    x = torch.randn(8, 16)
    y = net(x)
    assert y.shape == (8, 3)


def test_model_modules_importable():
    """Архитектуры моделей импортируются БЕЗ GPU/triton/MT5 (проверка переносимости)."""
    for mod in ["models.directional_net", "models.ensemble_net",
                "models.regime_net", "models.qnd_net"]:
        importlib.import_module(mod)


def test_demo_pipeline_runs():
    """Демо-пайплайн (синтетика -> признаки -> модель) отрабатывает."""
    import demo
    df = demo.make_synthetic_ohlc(500)
    feats = demo.causal_features(df)
    assert len(feats) > 0
    net = demo.TinyNet(feats.shape[1])
    out = net(torch.tensor(feats.values, dtype=torch.float32))
    assert out.shape[0] == len(feats)
