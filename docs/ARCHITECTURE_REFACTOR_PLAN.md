# План архитектурной переработки: Core, LangGraph, CRUD, LLM

## 1. Цели

1. **Telegram-бот** (`tg-module/bot.py`) остаётся транспортом: приём сообщений, кнопок, отображение UI; вся предметная логика уезжает в сервис **core**.
2. **core** реализует сценарии диалога как **ориентированный граф** на **LangGraph**: явные фазы, маршрутизация, возможность доработки (в т.ч. под «поиск вакансий» и новые ветки).
3. **Доступ к SQLite** только через отдельный **CRUD-сервис**; `model` (LLM-модуль) отвечает **только** за вызовы LLM API (чат, tool calls), без прямой работы с БД и без оркестрации RAG/бизнес-логики.

## 2. Текущее состояние (краткий разбор)

| Компонент | Сейчас |
|-----------|--------|
| `tg-module/bot.py` | Транспорт; HTTP в **core** (`LLM_BASE_URL` / совместимые пути `start_talk`, `clean_history`, профессии/роадмап); передаёт `parameters` (в т.ч. `test_results`, `user_display_name`). |
| `core/` | FastAPI + **LangGraph** (`core/graph_workflow.py`): `prepare` → **guard** (анти-инъекции) → `dialog` или короткое завершение → `persist`. Диалоговая логика в `core/dialog_model.py`; CRUD и LLM — только через HTTP-клиенты. |
| `crud-service/` | Единственная запись в SQLite через `repo/repository.py`; REST для сессии по **identity** и legacy-путь под Telegram. |
| `model/llm_service.py` | **llm-service**: `/v1/chat`, `/v1/tool_call`; без Repository. |
| `model/main.py` + `start_llm.py` | **Наследие** для локальной отладки и сравнения с прежним монолитом; в Docker по умолчанию не используется как единый API. |

Состояния пользователя в коде: `UserState` (`who`, `about`, `test`, `recommendation`, `talk`, служебный `inject_attempt` для атомарного persist при блокировке инъекции) — база для дальнейшего дробления графа.

## 3. Целевая топология сервисов

```
[Telegram Bot] --HTTP--> [core : LangGraph + оркестрация]
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
   [crud-service]      [llm-service]      [RAG / данные]
   (SQLite)            (chat, tools)      (FAISS, education JSON, …)
```

- **RAG** (поиск по профессиям/курсам) вызывается из **core** (как библиотека в том же контейнере на первом этапе или отдельный сервис позже). LLM-сервис к RAG не ходит.
- **Tavily / веб-поиск** — зависимость core (переменная `WEB_SEARCH_API_URL` в compose).

## 3.1. Данные и персистентность (реализовано)

- **Идентификация:** внутренний `app_user` + `user_identity` (`provider`, `external_user_id`). Legacy-путь **`/users/{id}/session`** трактует `id` как внешний id провайдера по умолчанию (`DEFAULT_IDENTITY_PROVIDER`, обычно `telegram`).
- **Нормализованные сущности:** `who_user`, `conversation` (и связанные таблицы метаданных/состояния диалога) с ключом `app_user_id`.
- **Docker:** SQLite файл на хосте через volume (например `./repo` → `/app/persist`, `SQLITE_PATH=/app/persist/test_app.sqlite3` у `crud-service`).
- **Миграция со старой схемы:** при старте репозитория для SQLite выполняется перенос legacy-таблиц (`user_metadata` / `conversation_history` без `app_user_id`) — см. `_sqlite_migrate_legacy_if_needed` в `repo/repository.py`. Для уже существующей `who_user` — отдельные `ALTER` под новые поля (`user_display_name`, `recommended_professions_json`).

## 4. LangGraph: состояния и узлы

### 4.1. Смысловые фазы (маппинг на текущий продукт)

| Фаза (логическая) | Соответствие в коде | Назначение |
|-------------------|---------------------|------------|
| **slot_extraction_who** | `UserState.WHO` | Сбор «кто пользователь», выход по маркеру `[EXIT]` → суммаризация, тип пользователя |
| **slot_extraction_about** | `UserState.ABOUT` | Уточнение контекста по типу (школьник / студент / работник) |
| **testing** | `UserState.TEST` | Рекомендация теста, v1 — текстовый тест в LLM; v2 — кнопки в Telegram + `test_results` в API |
| **recommendation_build** | `UserState.RECOMMENDATION` | Формирование подборки профессий (JSON + текст) |
| **free_talk** | `UserState.TALK` | Диалог с учётом подборки, детекция «новой рекомендации» |
| **profession_rag** | `go_rag` | «Поиск»/обогащение по выбранной профессии (RAG + LLM) |
| **roadmap_rag** | `go_rag_roadmap` | Роадмап обучения |
| **injection_handling** | `UserState.INJECT_ATTEMPT` (кратковременно) | Только для записи в CRUD: зафиксировать попытку и вернуть фазу; пользователю — безопасный ответ из графа |
| **fallback** | ошибки, таймауты | Единообразный ответ пользователю + логирование |

Примечание: формулировка «поиск вакансий» в ТЗ трактуется как **расширение**: отдельный узел/подграф (например `job_search`) можно добавить после выделения API вакансий (HH и т.д.) без ломки транспорта Telegram.

### 4.2. Граф turn-сценария (фактическая реализация)

Цепочка: **`prepare`** (гидратация из CRUD, опционально `user_display_name` из `parameters`) → **`guard`** (`core/injection_guard.py`: эвристика prompt-injection / jailbreak) → условие:

- при срабатывании guard — ответ-заглушка и ветка **`persist`** (двойной persist: зафиксировать `injection_attempts` в метаданных и восстановить прежнюю фазу);
- иначе — **`dialog`** (вызов `DialogModel.start_talk`) → **`persist`**.

Эндпоинты **профессии / роадмап** в core обходят этот граф и вызывают методы модели напрямую (guard там не дублируется — при необходимости вынести в общий слой).

### 4.3. Граф (цель — детализация E5)

- Условное ветвление от **типа входа**: обычное сообщение / `clean` / `profession_detail` / `roadmap` (частично уже на уровне отдельных HTTP-роутов).
- Внутри ветки «диалог» — **conditional edges** по `user_state` на узлы `handle_who`, `handle_about`, `handle_test`, `handle_recommendation`, `handle_talk`.
- После каждого успешного шага — обновление state и **persist** в CRUD (метаданные + при необходимости история).

## 5. API контракты

### 5.1. core (для бота и клиентов)

**Предпочтительные пути (v1):**

- `POST /v1/dialog/turn` — тело: `user_id`, `prompt`, `parameters` (как `Context`: в т.ч. `test_results`, `user_display_name`).
- `POST /v1/dialog/clean` — сброс сессии пользователя (совместимый ответ `LLMResponse`).
- `POST /v1/profession/info`, `POST /v1/profession/roadmap` — как `ProfessionRequest`.
- Ответ диалога: `LLMResponse` (`msg`, `professions`, `test_info`).

**Совместимость со старым ботом:** `POST /start_talk/`, `/clean_history/`, `/get_profession_info/`, `/get_profession_roadmap/` — прокси на v1-обработчики.

Прочее: `GET /health`, `GET /`, метрики Prometheus через `prometheus-fastapi-instrumentator`.

### 5.2. crud-service

**По identity (расширяемо под несколько провайдеров):**

- `GET/PUT /v1/identities/{provider}/{external_user_id}/session` — снимок сессии (`user_state`, `user_type`, `user_metadata`, `conversation_history`).
- `DELETE /v1/identities/{provider}/{external_user_id}` — очистка сессии.
- `DELETE /v1/identities/{provider}/{external_user_id}/full` — полное удаление пользователя.
- `GET /v1/identities/{provider}/{external_user_id}/app-user` — разрешение во внутренний id.

**Legacy (Telegram по умолчанию):**

- `GET/PUT /users/{user_id}/session`, `DELETE /users/{user_id}`.

### 5.3. llm-service

- `POST /v1/chat` — список сообщений `{role, text}` → `{text, token_usage?}`.
- `POST /v1/tool_call` — сообщение + имя набора tools из конфига.

Промпты и YAML с инструментами остаются в образе LLM-сервиса (или выносятся в volume — по мере рефакторинга).

## 6. Наблюдаемость

- **Логи:** stdout/stderr процесса (в Docker — `docker compose logs -f core` и аналогично для других сервисов). У **core** уровень задаётся `CORE_LOG_LEVEL` (см. `core/main.py`); формат через `logging.basicConfig` при отсутствии корневых handlers.
- **Метрики:** HTTP-метрики на core (и при необходимости дальнейшее распределение по сервисам).

## 7. Этапы внедрения

| Этап | Содержание |
|------|------------|
| **E0** | Документ (этот файл) |
| **E1** | CRUD-сервис + docker-compose; персистентный SQLite |
| **E2** | Выделение LLM HTTP API из `model`; удаление `Repository` из LLM-процесса |
| **E3** | Сервис `core` с LangGraph, HTTP для бота |
| **E4** | `tg-module`: запросы в `core` (в т.ч. совместимые пути) |
| **E5** | Расщепление узла `dialog` на отдельные LangGraph-ноды по `user_state`; явный persist после микро-шагов при необходимости |
| **E6** | (Опционально) отдельный RAG-микросервис; узел «вакансии» |

## 8. Риски и заметки

- **Состояние в памяти:** `DialogModel` может кэшировать данные между запросами; источник истины после шага — CRUD. При горизонтальном масштабировании нескольких реплик core потребуется внешний кэш/липкие сессии.
- **PROF_CONTEXT_DICT** в памяти — при масштабировании core вынести в кэш/БД (E5/E6).
- **Метрики Prometheus:** часть наследия остаётся в старом `start_llm.py`; целевое распределение — LLM-сервис (вызовы модели), core (бизнес-метрики).

## 9. Статус реализации

- [x] План в `docs/ARCHITECTURE_REFACTOR_PLAN.md`
- [x] E1: `crud-service`, схема `app_user` / `user_identity` / нормализованные таблицы, legacy-миграция SQLite, volume для БД
- [x] E2: `llm-service` (`model/llm_service.py`), только чат и tool_call
- [x] E3: `core` + LangGraph (`prepare` → `guard` → `dialog` / bypass → `persist`), `core/dialog_model.py`, клиенты CRUD/LLM
- [x] E4: compose + бот на `core`; доп. поля и UX (отображаемое имя, рекомендации в БД) по мере необходимости
- [x] Защита от prompt-injection на пути диалога (`core/injection_guard.py`, учёт в persist)
- [x] Базовое логирование core (`CORE_LOG_LEVEL`, stdout)
- [ ] E5: разбиение «толстой» ноды `dialog` на узлы по `user_state`
- [ ] E6: опционально RAG/вакансии как отдельные сервисы

**Наследие:** `model/main.py` и `model/start_llm.py` сохранены для локальной отладки и сравнения; в Docker по умолчанию используется связка `core` + `llm-service` + `crud-service`.
