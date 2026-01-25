#!/usr/bin/env python3
"""
Скрипт для запуска Telegram бота
"""
import asyncio
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, str(Path(__file__).parent))

from bot import main

if __name__ == "__main__":
    try:
        print("🚀 Запуск Telegram бота...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Бот остановлен пользователем")
    except Exception as e:
        print(f"❌ Ошибка при запуске бота: {e}")
        sys.exit(1)
