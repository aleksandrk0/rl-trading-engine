# RL Trading Engine (PyTorch)

![CI](https://github.com/aleksandrk0/rl-trading-engine/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Инженерный каркас системы алготрейдинга на обучении с подкреплением: ансамбль специализированных нейросетей для торговых решений, риск-менеджмент, асинхронный брокерский слой, кастомные GPU-ядра и тесты.

> ⚠️ **Это витрина инженерии, а не торговая стратегия.** Здесь показаны архитектуры моделей, инфраструктура и методология валидации. Торговые сигналы, обученные веса, параметры стратегии и реальные данные **намеренно не публикуются** (`.gitignore`). Метрики прибыльности тоже не заявляются — почему, см. раздел «Методология».

## Быстрый старт

```bash
git clone https://github.com/aleksandrk0/rl-trading-engine.git
cd rl-trading-engine
pip install -r requirements.txt        # только ядро, ставится на любой ОС (CPU)

pytest tests/test_smoke.py -v          # smoke-тест: всё импортируется и работает
python demo.py                         # синтетика -> признаки -> модель -> demo_equity.png
python validate.py                     # shuffle-label + walk-forward (каркас валидации)
```
Ни ключей, ни MetaTrader5, ни GPU, ни реальных данных для этого не нужно.

## Архитектура

| Модуль | Назначение |
|---|---|
| `models/` | `feature_net` (извлечение признаков + опц. GPU-ядра на triton), `directional_net` (направление, LOB-анализ, conv1d), `regime_net` (режим рынка), `qnd_net`, `ensemble_net` (ансамблирование), `trade_recorder` |
| `trading/` | `broker.py` — асинхронный брокерский слой с переподключением; `risk_manager.py` — риск, позиции, лимиты |
| `utils/` | `monitoring.py` — мониторинг метрик |
| `tools/` | `download_history.py` — загрузка истории (MT5, Windows) |
| `tests/` | `test_smoke.py` (переносимость), тесты моделей/размерностей/препроцессинга; `mock_mt5.py` |
| `demo.py` · `validate.py` | самодостаточные демо и валидация на синтетике |

## Методология и метрики (честно)

Репозиторий демонстрирует **инженерию**, а не торговый перформанс. Числа прибыльности/точности как «достижение» не публикуются по двум причинам: (1) это edge стратегии; (2) единичная цифра бэктеста вводит в заблуждение.

**Что сделано корректно (проверяемо по коду):**
- **Хронологический** train/val split (по времени), не случайный → нет утечки от перемешивания ряда.
- **Каузальные признаки**: индикаторы через `talib` (трейлинговые), без `center=True` и без `shift(-N)` в фичах.
- **Нормализация по батчу** (не по всей серии) → нет утечки статистик из val в train.

**Про точность.** Промежуточная directional accuracy на hold-out выглядит высокой, но я **не указываю её как результат** до подтверждения робастности: один прогон ≠ результат. Корректная оценка — **walk-forward на нескольких окнах** (распределение: медиана + разброс) плюс **shuffle-label тест** на отсутствие утечки. Каркас обеих проверок — в [`validate.py`](validate.py).

## Переносимость / другое железо

- **CPU-fallback** встроен: `device = cuda if torch.cuda.is_available() else cpu`.
- **GPU-ядра (triton)** — опциональны (импорт под `try/except`, проверка `is_triton_available()`); без NVIDIA/Linux код импортируется и работает на CPU-пути.
- **Windows-only** (MetaTrader5, pywin32) и тяжёлое — в [`requirements-live.txt`](requirements-live.txt), ядро в [`requirements.txt`](requirements.txt) ставится везде.
- **Docker** (Linux, CPU): `docker build -t rl-trading-engine . && docker run --rm rl-trading-engine` — прогонит тесты и демо.
- **CI** (GitHub Actions) гоняет smoke-тест и демо на Ubuntu / Python 3.10–3.11.

## Стек

PyTorch · NumPy · pandas · Triton (опц., GPU-ядра) · TA-Lib · MetaTrader5 / ccxt / websockets (опц.) · asyncio · pytest · Docker · GitHub Actions

---
*При необходимости заполните раздел метрик после walk-forward.*
