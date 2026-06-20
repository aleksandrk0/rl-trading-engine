# Воспроизводимая среда (Linux, CPU) — доказательство «работает на другом железе».
# Сборка:  docker build -t rl-trading-engine .
# Запуск:  docker run --rm rl-trading-engine        # прогонит тесты и демо
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# По умолчанию: тесты + демо (всё на синтетике, без ключей и GPU)
CMD ["sh", "-c", "pytest tests/test_smoke.py -q && python demo.py"]
