"""
Ранее здесь был монолитный API поверх ``start_llm.Model`` (удалён).

Диалог и состояние — **core** (``POST /v1/dialog/turn`` и др.).
Вызовы LLM — **llm-service** (``POST /v1/chat``, ``POST /v1/tool_call``).

Этот модуль оставлен только чтобы не ломать ``model/app_run.py`` и старые ссылки на ``model.main:app``:
все бывшие эндпоинты отвечают **410 Gone**.
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from model.shema import Context, ProfessionRequest

app = FastAPI(title="Legacy model API (removed)", version="0.0.0")
Instrumentator().instrument(app).expose(app)

_GONE = {
    "detail": (
        "Монолитный API (start_llm) удалён. Используйте сервис **core** "
        "(например POST /v1/dialog/turn) и **llm-service**."
    )
}


@app.get("/")
def read_root():
    return {
        "message": "Монолитный Model API отключён; используйте core + llm-service + crud.",
        "removed": "model/start_llm.py",
    }


def _gone():
    return JSONResponse(status_code=410, content=_GONE)


@app.post("/start_talk/")
async def predict(_context: Context):
    return _gone()


@app.post("/get_user_info/")
async def get_user_info(_context: Context):
    return _gone()


@app.post("/get_profession_info/")
async def get_profession_info(_request: ProfessionRequest):
    return _gone()


@app.post("/get_profession_roadmap/")
async def get_profession_roadmap(_request: ProfessionRequest):
    return _gone()


@app.post("/clean_history/")
async def clean_history(_context: Context):
    return _gone()


@app.post("/finish_test_early/")
async def finish_test_early(_context: Context):
    return _gone()
