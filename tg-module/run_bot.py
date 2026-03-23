#!/usr/bin/env python3
"""
Скрипт для запуска Telegram бота
"""
import asyncio
import sys
from pathlib import Path

# Корень репозитория — для common.*; tg-module — для bot и локальных импортов
_tg_dir = Path(__file__).resolve().parent
_repo_root = _tg_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_tg_dir))

import logging

from bot import main

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    try:
        print("🚀 Запуск Telegram бота...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Бот остановлен пользователем")
    except Exception as e:
        log.exception("Ошибка при запуске бота: %s", e)
        print(f"❌ Ошибка при запуске бота: {e}")
        sys.exit(1)
