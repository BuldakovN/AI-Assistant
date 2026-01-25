# Парсеры вакансий и суммаризация

В этой директории находятся скрипты для сбора данных о профессиях и их суммаризации

## Парсер hh.ru (`hh.py`)

Скрипт последовательно обходит вакансии на `hh.ru`, группирует их в профессии по схожести и сохраняет два файла:
- `data/hh_detailed_professions.json` — подробные данные по профессиям с примерами вакансий
- `data/hh_professions_comparison.json` — плоские тексты для последующей суммаризации

### Переменные окружения
Скрипт читает значения из `.env` (используется `python-dotenv`). В корне проекта создайте файл `.env` со значениями:

```bash
HH_CLIENT_ID=ваш_client_id
HH_CLIENT_SECRET=ваш_client_secret
HH_REDIRECT_URI=ваш_redirect_uri
```

### Настройки парсинга
Меняются прямо в начале `parsers/hh.py`:
- `FUZZY_MATCH_THRESHOLD` — порог схожести для отнесения вакансии к профессии (0–100)
- `MAX_PROFESSIONS` — сколько топ-профессий вернуть в результате
- `MAX_PAGES` — сколько страниц обходить (лимит HH до 2000)
- `VACANCIES_PER_PAGE` — вакансий на страницу (лимит HH до 100)
- `USE_TFIDF`, `TFIDF_MAX_FEATURES`, `TOP_SKILLS` — параметры TF‑IDF улучшения сравнения

### Запуск
Из корня репозитория:

```bash
python -m parsers.hh
```

После завершения в каталоге `data/` появятся два файла с результатами (см. выше).

## Суммаризатор профессий (`summary_profession.py`)

Берёт входной файл `data/hh_professions_comparison.json` и генерирует:
- `data/summary_hh_professions_detailed.json` — структурированное описание профессий
- `data/summary_hh_professions_comparison.json` — плоские тексты (агрегированные)

### Переменные окружения
Скрипт также читает `.env`:

```bash
# Режим через ваш локальный/удалённый API (по умолчанию)
LLM_API_URL=http://localhost:8000/start_talk/
LLM_SUMMARIZER_USER_ID=summarizer

# Для прямого вызова Yandex Foundation Models (опционально)
YANDEX_CLOUD_FOLDER=
YANDEX_CLOUD_API_KEY=
YANDEX_MODEL=yandexgpt
```

- По умолчанию используется режим `api` (отправка в `LLM_API_URL`).
- Можно переключиться на прямой режим `direct` и/или включить фоллбек на прямой вызов.

### Запуск
Из корня репозитория:

```bash
python -m parsers.summary_profession --input data/hh_professions_comparison.json \
  --limit 100 \
  --concurrency 3 \
  --resume \
  --llm-mode api
```

Параметры:
- `--input` — входной JSON с плоскими текстами (по умолчанию `data/hh_professions_comparison.json`)
- `--limit` — ограничить количество профессий на выход
- `--concurrency` — количество параллельных запросов
- `--resume` — дозаполнять существующий файл результатов
- `--llm-mode` — `api` (по умолчанию) или `direct`
