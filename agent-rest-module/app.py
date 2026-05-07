from __future__ import annotations

import logging
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from common.error_logging import setup_service_error_logging
from llm_client import CoreApiClient, LLMResponse
from models import (
    AgentActionRequest,
    AgentContext,
    AgentMessageRequest,
    AgentResponse,
    AgentSessionSnapshot,
    AgentStartRequest,
    ProfessionRequest,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
setup_service_error_logging("agent-rest-module")

app = FastAPI(
    title="Agent REST Module",
    description="REST adapter for automated agent-to-core interaction.",
    version="0.1.0",
)


def _extract_top5(professions: Dict[str, object]) -> list[str]:
    return list(professions.keys())[:5] if professions else []


def _to_response(agent_id: str, context: AgentContext, llm_response: LLMResponse) -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        message=llm_response.msg,
        user_state=llm_response.user_state,
        professions=llm_response.professions,
        test_info=llm_response.test_info,
        test_version=llm_response.test_version,
        top5_professions=_extract_top5(llm_response.professions),
        context=context,
    )


@app.get("/")
async def root() -> dict:
    return {"message": "Agent REST module is running"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/sessions/start", response_model=AgentResponse)
async def start_session(payload: AgentStartRequest) -> AgentResponse:
    try:
        async with CoreApiClient() as client:
            resp = await client.generate_response(
                payload.initial_prompt,
                payload.agent_id,
                parameters=payload.parameters,
            )
        return _to_response(payload.agent_id, payload.context, resp)
    except Exception as exc:
        logger.exception("Failed to start session for agent_id=%s", payload.agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/sessions/{agent_id}/message", response_model=AgentResponse)
async def session_message(agent_id: str, payload: AgentMessageRequest) -> AgentResponse:
    try:
        async with CoreApiClient() as client:
            resp = await client.generate_response(
                payload.message,
                agent_id,
                parameters=payload.parameters,
            )
        return _to_response(agent_id, payload.context, resp)
    except Exception as exc:
        logger.exception("Failed message call for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/sessions/{agent_id}/finish-test", response_model=AgentResponse)
async def finish_test(agent_id: str, payload: AgentActionRequest) -> AgentResponse:
    try:
        async with CoreApiClient() as client:
            resp = await client.finish_test_early(agent_id, parameters=payload.parameters)
        return _to_response(agent_id, payload.context, resp)
    except Exception as exc:
        logger.exception("Failed finish-test call for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/sessions/{agent_id}/clean-history")
async def clean_history(agent_id: str, payload: AgentActionRequest) -> dict:
    try:
        async with CoreApiClient() as client:
            result = await client.clean_user_history(agent_id, parameters=payload.parameters)
        return {
            "agent_id": agent_id,
            "result": result,
            "context": payload.context.model_dump(),
        }
    except Exception as exc:
        logger.exception("Failed clean-history call for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/sessions/{agent_id}/profession-info", response_model=AgentResponse)
async def profession_info(agent_id: str, payload: ProfessionRequest) -> AgentResponse:
    try:
        async with CoreApiClient() as client:
            resp = await client.get_profession_info(agent_id, payload.profession_name)
        return _to_response(agent_id, payload.context, resp)
    except Exception as exc:
        logger.exception("Failed profession-info call for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/sessions/{agent_id}/profession-roadmap", response_model=AgentResponse)
async def profession_roadmap(agent_id: str, payload: ProfessionRequest) -> AgentResponse:
    try:
        async with CoreApiClient() as client:
            resp = await client.get_profession_roadmap(agent_id, payload.profession_name)
        return _to_response(agent_id, payload.context, resp)
    except Exception as exc:
        logger.exception("Failed profession-roadmap call for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/sessions/{agent_id}", response_model=AgentSessionSnapshot)
async def get_session(agent_id: str) -> AgentSessionSnapshot:
    try:
        async with CoreApiClient() as client:
            session = await client.get_user_session(agent_id)
        return AgentSessionSnapshot(
            agent_id=agent_id,
            app_user_id=session.get("app_user_id"),
            user_state=session.get("user_state"),
            user_type=session.get("user_type"),
            conversation_history_len=len(session.get("conversation_history") or []),
            user_metadata=session.get("user_metadata") or {},
        )
    except Exception as exc:
        logger.exception("Failed to fetch session for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

