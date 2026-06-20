# tools/download_history.py
# Утилита загрузки тиков из MetaTrader5 (только Windows + установленный MT5).
# Импорт опциональный, чтобы пакет импортировался на любой ОС.

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # MT5 доступен только на Windows; на других ОС утилита недоступна

import pandas as pd
from datetime import datetime, timedelta
import pytz

def download_and_format_ticks(symbol="EURUSD", days=7, filename="eurusd_ticks.csv"):
    """
    Скачивает и форматирует тики в правильный формат
    
    Args:
        symbol: Торговый символ
        days: Количество дней истории
        filename: Имя файла для сохранения
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 не установлен (доступен только на Windows). "
                           "Установите: pip install -r requirements-live.txt")
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    # Установка временной зоны
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime.now(tz=timezone) - timedelta(days=days)

    # Получение тиков
    ticks = mt5.copy_ticks_from(symbol, utc_from, 100000, mt5.COPY_TICKS_ALL)
    
    # Конвертация в pandas
    df = pd.DataFrame(ticks)
    
    # Форматирование времени
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Форматирование данных
    formatted_df = pd.DataFrame({
        'timestamp': df['time'],
        'ask': df['ask'].round(5),
        'bid': df['bid'].round(5),
        'high': df[['ask', 'bid']].max(axis=1).round(5),
        'low': df[['ask', 'bid']].min(axis=1).round(5),
        'volume': df['volume_real'],
        'last': df['last'].fillna(df['bid']).round(5),
        'spread': (df['ask'] - df['bid']).round(5)
    })

    # Сохранение в CSV
    formatted_df.to_csv(f'data/historical/{filename}', index=False, float_format='%.5f')
    print(f"Saved {len(formatted_df)} formatted ticks to {filename}")

    mt5.shutdown()

# Пример правильного формата:
"""
timestamp,ask,bid,high,low,volume,last,spread
2024-01-04 00:00:00.123,1.04259,1.04258,1.04259,1.04258,1.0,1.04258,0.00001
2024-01-04 00:00:00.456,1.04260,1.04259,1.04260,1.04259,1.5,1.04259,0.00001
"""

if __name__ == '__main__':
    download_and_format_ticks()