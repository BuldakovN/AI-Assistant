#!/usr/bin/env python3
"""
Запуск заглушки legacy API на :8000 (``model.main:app``).

Монолитный диалог (``start_llm``) удалён; POST-эндпоинты отвечают 410.
Для реального бота используйте **core** и **llm-service**.
"""
import sys
from pathlib import Path

import uvicorn

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("Запуск legacy model API (заглушка)...")
    print("http://localhost:8000 — только GET / и 410 на старые POST.")
    print("Документация: http://localhost:8000/docs")
    print("Остановка: Ctrl+C")

    try:
        uvicorn.run(
            "model.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
    except KeyboardInterrupt:
        print("\nAPI остановлен")
    except Exception as e:
        print(f"Ошибка при запуске API: {e}")
        sys.exit(1)
