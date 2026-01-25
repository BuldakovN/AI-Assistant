## Векторный поиск профессий

Эта директория содержит инструменты для построения и использования FAISS-индекса профессий на основе эмбеддингов Яндекс (YandexGPT Embeddings). Индекс строится по агрегированному файлу `data/summary_hh_professions_comparison.json` и сохраняется локально для быстрого семантического поиска.

### Состав
- `build_faiss_index.py` — сборка FAISS-индекса по данным `summary_hh_professions_comparison.json`.
- `search_professions.py` — пример поиска по готовому индексу (kNN, топ-K результатов).
- `yandex_embeddings.py` — обёртка над YandexGPT Embeddings (получение эмбеддингов для текстов).
- `docs.json` — карта документов (метаданные), сохраняется после сборки индекса.
- `index.faiss`, `index.pkl` — бинарные артефакты индекса FAISS/метаданные векторного стора.


### Переменные окружения
Перед запуском укажите переменные окружения (можно в `.env`):
- `YANDEX_CLOUD_API_KEY` — API-ключ доступа к Yandex Cloud
- `YANDEX_CLOUD_FOLDER` — ID каталога (folder) в Yandex Cloud

Формат `.env`:
```
YANDEX_CLOUD_API_KEY=...
YANDEX_CLOUD_FOLDER=...
```

### Установка зависимостей
В корне проекта:
```
pip install -r requirements.txt
```

### Источник данных
Индекс строится по файлу:
```
data/summary_hh_professions_comparison.json
```
Каждая запись агрегирует информацию о профессии. Текст берётся как конкатенация полей или ключ, если текст пустой.

### Сборка индекса
Запустите скрипт сборки индекса:
```
python -m professions_vector_index.build_faiss_index
```
Скрипт:
- загрузит `.env`
- подготовит тексты и метаданные
- сделает эмбеддинги батчами (по 5, с задержкой ~1.2 c для соблюдения лимитов API)
- создаст FAISS-индекс и сохранит его в директорию `data/profession_vector` (`index.faiss`, `index.pkl`, `docs.json`)

Ожидаемый вывод в конце:
```
Индекс сохранён в: data/profession_vector
```

### Поиск по индексу
После сборки можно выполнять поиск семантически близких профессий. Пример (если в `search_professions.py` предусмотрен CLI):
```
python -m professions_vector_index.search_professions --query "Data Scientist" --k 5
```
Либо импортируйте модуль в своём коде и используйте методы FAISS/VectorStore из LangChain.

### Обновление индекса
Если данные изменились (обновился `summary_hh_professions_comparison.json`), перезапустите сборку:
```
python -m professions_vector_index.build_faiss_index
```