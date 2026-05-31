# Agent REST Module

REST-адаптер для агентного тестирования профориентационного сценария без Telegram.

## Основные эндпоинты

- `POST /v1/sessions/start` — запуск сессии агента
- `POST /v1/sessions/{agent_id}/message` — шаг диалога
- `POST /v1/sessions/{agent_id}/finish-test` — принудительное завершение теста
- `POST /v1/sessions/{agent_id}/clean-history` — очистка истории
- `POST /v1/sessions/{agent_id}/profession-info` — детальная информация о профессии
- `POST /v1/sessions/{agent_id}/profession-roadmap` — роудмап профессии
- `GET /v1/sessions/{agent_id}` — снимок сессии из CRUD

## Локальный запуск

```bash
pip install -r agent-rest-module/requirements.txt
python agent-rest-module/run_server.py
```

## Переменные окружения

- `LLM_API_URL` (default: `http://localhost:8020/start_talk/`)
- `LLM_FINISH_TEST_URL` (default: `http://localhost:8020/finish_test_early/`)
- `LLM_PROFESSION_API_URL` (default: `http://localhost:8020/get_profession_info/`)
- `LLM_ROADMAP_API_URL` (default: `http://localhost:8020/get_profession_roadmap/`)
- `LLM_CLEAN_API_URL` (default: `http://localhost:8020/clean_history/`)
- `CRUD_SERVICE_URL` (default: `http://localhost:8010`)
- `AGENT_REST_HOST` (default: `0.0.0.0`)
- `AGENT_REST_PORT` (default: `8040`)
