from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp


logger = logging.getLogger(__name__)


def _client_timeout() -> aiohttp.ClientTimeout:
    """HTTP timeouts for core LLM calls (full chain can take several minutes)."""
    try:
        total = float(os.getenv("CORE_HTTP_TIMEOUT_TOTAL_SECONDS", "600"))
    except ValueError:
        total = 600.0
    try:
        connect = float(os.getenv("CORE_HTTP_TIMEOUT_CONNECT_SECONDS", "30"))
    except ValueError:
        connect = 30.0
    return aiohttp.ClientTimeout(total=total, connect=connect)


@dataclass
class LLMResponse:
    msg: str
    professions: Dict[str, Any] = field(default_factory=dict)
    test_info: Dict[str, Any] = field(default_factory=dict)
    user_state: Optional[str] = None
    test_version: Optional[str] = None


class CoreApiClient:
    def __init__(self) -> None:
        self.api_url = os.getenv("LLM_API_URL", "http://localhost:8020/start_talk/")
        self.profession_api_url = os.getenv(
            "LLM_PROFESSION_API_URL", "http://localhost:8020/get_profession_info/"
        )
        self.roadmap_api_url = os.getenv(
            "LLM_ROADMAP_API_URL", "http://localhost:8020/get_profession_roadmap/"
        )
        self.clean_api_url = os.getenv("LLM_CLEAN_API_URL", "http://localhost:8020/clean_history/")
        self.finish_test_early_url = os.getenv(
            "LLM_FINISH_TEST_URL", "http://localhost:8020/finish_test_early/"
        )
        self.crud_base_url = os.getenv("CRUD_SERVICE_URL", "http://localhost:8010").rstrip("/")
        self.timeout = _client_timeout()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "CoreApiClient":
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session:
            await self.session.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("CoreApiClient not initialized, use async with")
        return self.session

    async def _post_llm_response(self, url: str, payload: Dict[str, Any], endpoint_name: str) -> LLMResponse:
        session = self._ensure_session()
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                detail = await response.text()
                logger.error("Core endpoint=%s failed status=%s detail=%s", endpoint_name, response.status, detail)
                raise RuntimeError(f"{endpoint_name} returned {response.status}: {detail}")
            result = await response.json()
            return LLMResponse(
                msg=result.get("msg", ""),
                professions=result.get("professions") or {},
                test_info=result.get("test_info") or {},
                user_state=result.get("user_state"),
                test_version=result.get("test_version"),
            )

    async def generate_response(self, message: str, agent_id: str, parameters: Optional[Dict[str, Any]] = None) -> LLMResponse:
        payload = {
            "user_id": str(agent_id),
            "prompt": message,
            "parameters": parameters or {},
        }
        return await self._post_llm_response(self.api_url, payload, "start_talk")

    async def finish_test_early(self, agent_id: str, parameters: Optional[Dict[str, Any]] = None) -> LLMResponse:
        payload = {
            "user_id": str(agent_id),
            "prompt": "",
            "parameters": parameters or {},
        }
        return await self._post_llm_response(self.finish_test_early_url, payload, "finish_test_early")

    async def clean_user_history(self, agent_id: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        session = self._ensure_session()
        payload = {
            "user_id": str(agent_id),
            "prompt": "",
            "parameters": parameters or {},
        }
        async with session.post(self.clean_api_url, json=payload) as response:
            if response.status != 200:
                detail = await response.text()
                logger.error("Core endpoint=clean_history failed status=%s detail=%s", response.status, detail)
                raise RuntimeError(f"clean_history returned {response.status}: {detail}")
        return "history_cleared"

    async def get_profession_info(self, agent_id: str, profession_name: str) -> LLMResponse:
        payload = {
            "user_id": str(agent_id),
            "profession_name": profession_name,
        }
        return await self._post_llm_response(self.profession_api_url, payload, "get_profession_info")

    async def get_profession_roadmap(self, agent_id: str, profession_name: str) -> LLMResponse:
        payload = {
            "user_id": str(agent_id),
            "profession_name": profession_name,
        }
        return await self._post_llm_response(self.roadmap_api_url, payload, "get_profession_roadmap")

    async def get_user_session(self, agent_id: str) -> Dict[str, Any]:
        session = self._ensure_session()
        url = f"{self.crud_base_url}/users/{agent_id}/session"
        async with session.get(url) as response:
            if response.status != 200:
                detail = await response.text()
                logger.error("CRUD session lookup failed status=%s detail=%s", response.status, detail)
                raise RuntimeError(f"session lookup returned {response.status}: {detail}")
            return await response.json()

