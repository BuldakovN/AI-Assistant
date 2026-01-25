# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем код модуля
COPY tg-module/ ./tg-module/

# Копируем файл зависимостей
#COPY tg-module/requirements.txt ./requirements.txt

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r tg-module/requirements.txt


# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Команда для запуска бота
CMD ["python", "tg-module/run_bot.py"]
