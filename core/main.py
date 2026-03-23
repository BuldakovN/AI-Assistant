"""
Core API: оркестрация для Telegram и других клиентов.
"""
import logging
import os
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional

from prometheus_fastapi_instrumentator import Instrumentator

from common.error_logging import setup_service_error_logging
from core.dialog_model import DialogModel
from core.graph_workflow import build_turn_graph

_log_level = getattr(logging, (os.getenv("CORE_LOG_LEVEL", "INFO").upper()), logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logging.getLogger("core").setLevel(_log_level)
setup_service_error_logging("core")

app = FastAPI(title="Core Service", version="0.1.0")
Instrumentator().instrument(app).expose(app)

_dialog = DialogModel()
_turn_graph = build_turn_graph(_dialog)


class LLMResponse(BaseModel):
    msg: str
    professions: Optional[Dict[str, str]] = None
    test_info: Optional[Dict[str, Any]] = None


class Context(BaseModel):
    user_id: str
    prompt: str
    parameters: dict = {}


class ProfessionRequest(BaseModel):
    user_id: str
    profession_name: str


class Message(BaseModel):
    user_id: str
    msg: str
    timestamp: str


@app.get("/")
def read_root():
    return {"message": "Welcome to Core API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/dialog/turn", response_model=LLMResponse)
async def dialog_turn(context: Context) -> LLMResponse:
    out = await _turn_graph.ainvoke(
        {
            "user_id": context.user_id,
            "prompt": context.prompt,
            "parameters": context.parameters,
        }
    )
    return LLMResponse(
        msg=out.get("response_msg") or "",
        professions=out.get("professions"),
        test_info=out.get("test_info"),
    )


@app.post("/v1/dialog/clean", response_model=LLMResponse)
async def dialog_clean(context: Context) -> LLMResponse:
    await _dialog.clean_user_history(context.user_id)
    return LLMResponse(
        msg=f"История общения и все метаданные о пользователе {context.user_id} удалены",
        professions=None,
        test_info=None,
    )


@app.post("/v1/profession/info", response_model=LLMResponse)
async def profession_info(request: ProfessionRequest) -> LLMResponse:
    await _dialog._hydrate_from_crud(request.user_id)
    response = await _dialog.go_rag(profession_name=request.profession_name, user_id=request.user_id)
    user_metadata = _dialog.user_metadata.get(request.user_id, {})
    ai_recommendation_json = user_metadata.get("ai_recommendation_json")
    professions = ai_recommendation_json["professions"] if ai_recommendation_json else None
    await _dialog.persist(request.user_id)
    return LLMResponse(msg=response, professions=professions)


@app.post("/v1/profession/roadmap", response_model=LLMResponse)
async def profession_roadmap(request: ProfessionRequest) -> LLMResponse:
    await _dialog._hydrate_from_crud(request.user_id)
    response = await _dialog.go_rag_roadmap(profession_name=request.profession_name, user_id=request.user_id)
    user_metadata = _dialog.user_metadata.get(request.user_id, {})
    ai_recommendation_json = user_metadata.get("ai_recommendation_json")
    professions = ai_recommendation_json["professions"] if ai_recommendation_json else None
    await _dialog.persist(request.user_id)
    return LLMResponse(msg=response, professions=professions)


@app.post("/get_user_info/")
async def get_user_info(context: Context):
    await _dialog._hydrate_from_crud(context.user_id)
    response = _dialog.get_user_info(user_id=context.user_id)
    return Message(user_id=context.user_id, msg=str(response), timestamp=str(datetime.now()))


# Совместимость со старыми путями бота (опционально)
@app.post("/start_talk/", response_model=LLMResponse)
async def start_talk_compat(context: Context) -> LLMResponse:
    return await dialog_turn(context)


@app.post("/clean_history/", response_model=LLMResponse)
async def clean_history_compat(context: Context) -> LLMResponse:
    return await dialog_clean(context)


@app.post("/get_profession_info/", response_model=LLMResponse)
async def get_profession_info_compat(request: ProfessionRequest) -> LLMResponse:
    return await profession_info(request)


@app.post("/get_profession_roadmap/", response_model=LLMResponse)
async def get_profession_roadmap_compat(request: ProfessionRequest) -> LLMResponse:
    return await profession_roadmap(request)
