"""
Оркестрация диалога. LLM и БД — через HTTP (llm-service, crud-service).
Фазы WHO → ABOUT → TEST (v1 или v2) → RECOMMENDATION → TALK.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from common.professions_normalize import normalize_ai_recommendation_json, normalize_professions
from core.config_data import config, web_search_api_url
from core.constants import UserState, UserType
from core.rag_professions import profession_rag
from core.test_flow_v1 import TestFlowV1
from core.test_flow_v2 import TestFlowV2

load_dotenv()

logger = logging.getLogger(__name__)

# Обратная совместимость: раньше импортировали из dialog_model
__all__ = ["DialogModel", "UserState", "UserType"]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _attach_profession_order(blob: Any) -> Any:
    if not isinstance(blob, dict):
        return blob
    prof = blob.get("professions")
    if isinstance(prof, dict) and prof:
        out = dict(blob)
        out["profession_order"] = list(prof.keys())
        return out
    return blob


class DialogModel:
    def __init__(self) -> None:
        self._crud_base = os.getenv("CRUD_SERVICE_URL", "http://localhost:8010").rstrip("/")
        self._llm_base = os.getenv("LLM_SERVICE_URL", "http://localhost:8001").rstrip("/")
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.user_state: Dict[str, str] = {}
        self.user_type: Dict[str, Optional[str]] = {}
        self.user_metadata: Dict[str, Dict[str, Any]] = {}
        self.test_variant = (os.getenv("test_run_version") or "v2").strip()
        self.skip_test_v1 = _env_flag("SKIP_TEST_V1_TO_RECOMMENDATION", default=False)
        self.user_last_seen: Dict[str, datetime] = {}
        self._web_search = None
        self._test_v1 = TestFlowV1()
        self._test_v2 = TestFlowV2()

    def set_user_state(self, user_id: str, new_state: str, reason: str) -> None:
        """Пишет ``user_state`` и логирует смену фазы (если значение изменилось)."""
        old = self.user_state.get(user_id)
        if old == new_state:
            return
        self.user_state[user_id] = new_state
        logger.info(
            "dialog_user_state_change user_id=%s old_state=%r new_state=%r reason=%s",
            user_id,
            old,
            new_state,
            reason,
        )

    def _web_search_client(self):
        if self._web_search is None:
            from core.web_search_travily import WebSearch

            self._web_search = WebSearch(api_key="asd", api_base_url=web_search_api_url)
        return self._web_search

    async def _hydrate_from_crud(self, user_id: str) -> None:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(f"{self._crud_base}/users/{user_id}/session")
            r.raise_for_status()
            data = r.json()
        self.conversation_history[user_id] = list(data.get("conversation_history") or [])
        self.set_user_state(
            user_id,
            data.get("user_state") or UserState.WHO,
            "hydrate_from_crud",
        )
        self.user_type[user_id] = data.get("user_type")
        self.user_metadata[user_id] = dict(data.get("user_metadata") or {})
        blob = self.user_metadata[user_id].get("ai_recommendation_json")
        if blob is not None:
            self.user_metadata[user_id]["ai_recommendation_json"] = normalize_ai_recommendation_json(blob)

    async def persist(self, user_id: str) -> None:
        payload = {
            "user_state": self.user_state.get(user_id, UserState.WHO),
            "user_type": self.user_type.get(user_id),
            "user_metadata": self.user_metadata.get(user_id, {}),
            "conversation_history": self.conversation_history.get(user_id, []),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.put(f"{self._crud_base}/users/{user_id}/session", json=payload)
            r.raise_for_status()

    async def clean_user_history(self, user_id: str) -> None:
        prev_state = self.user_state.get(user_id)
        self.conversation_history.pop(user_id, None)
        self.user_state.pop(user_id, None)
        if prev_state is not None:
            logger.info(
                "dialog_user_state_change user_id=%s old_state=%r new_state=<removed> reason=clean_user_history",
                user_id,
                prev_state,
            )
        self.user_type.pop(user_id, None)
        self.user_metadata.pop(user_id, None)
        self.user_last_seen.pop(user_id, None)
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.delete(f"{self._crud_base}/users/{user_id}")
            r.raise_for_status()

    def get_user_info(self, user_id: str) -> dict:
        return {
            user_id: {
                "user_state": self.user_state.get(user_id),
                "user_type": self.user_type.get(user_id),
                "user_metadata": self.user_metadata.get(user_id),
            }
        }

    async def add_system_message(self, content: str, user_id: str) -> None:
        self.conversation_history[user_id].append({"role": "system", "text": content})

    async def add_human_message(self, content: str, user_id: str) -> None:
        self.conversation_history[user_id].append({"role": "user", "text": content})

    async def add_ai_message(self, content: str, user_id: str) -> None:
        self.conversation_history[user_id].append({"role": "assistant", "text": content})

    async def _llm_chat_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(f"{self._llm_base}/v1/chat", json={"messages": messages})
            r.raise_for_status()
            data = r.json()
        return data["text"], int(data.get("tokens") or 0)

    async def extract_user_type(self, user_id: str) -> Optional[str]:
        who_user = self.user_metadata[user_id]["who_user"]
        user_type = await self.tool_run(who_user, tool_name="tools")
        ut = user_type["user_type"]
        if ut == UserType.SCHOOL:
            return UserType.SCHOOL
        if ut == UserType.STUDENT:
            return UserType.STUDENT
        if ut == UserType.WORKER:
            return UserType.WORKER
        return None

    async def summarization(self, user_id: str) -> str:
        story = ""
        for i in self.conversation_history[user_id]:
            if i["role"] == "system":
                continue
            story += f"{i['role']}: {i['text']}\n\n"

        prompt = config["summarization"]
        messages = [{"role": "system", "text": prompt}, {"role": "user", "text": story}]
        res, _llm_tokens = await self._llm_chat_messages(messages)
        return res

    async def update_user_state(self, ai_response: str, user_id: str) -> bool:
        current_state = self.user_state[user_id]
        pattern = r"\[EXIT]"
        match = re.search(pattern, ai_response, re.IGNORECASE)
        if match:
            if current_state == UserState.WHO and self.user_type[user_id] is None:
                user_story = await self.summarization(user_id)
                self.user_metadata[user_id] = {"who_user": user_story}
                self.set_user_state(user_id, UserState.ABOUT, "update_user_state_exit_marker_who")
                self.user_type[user_id] = await self.extract_user_type(user_id)
                return True

            if current_state == UserState.ABOUT:
                user_story = await self.summarization(user_id)
                self.user_metadata[user_id]["about_user"] = user_story
                self.set_user_state(user_id, UserState.TEST, "update_user_state_exit_marker_about")
                return True

            if current_state == UserState.TEST:
                self.user_metadata[user_id]["test_user"] = ai_response
                if self.test_variant == "v1":
                    user_story = await self.summarization(user_id)
                    self.user_metadata[user_id]["test_user"] = user_story
                self.set_user_state(
                    user_id,
                    UserState.RECOMMENDATION,
                    "update_user_state_exit_marker_test",
                )
                return True

            if current_state == UserState.TALK:
                return True

        return False

    async def tool_run(self, message: str, tool_name: str) -> Any:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{self._llm_base}/v1/tool_call",
                json={
                    "message": message,
                    "tool_key": tool_name,
                    "temperature": 0.6,
                    "max_tokens": 2000,
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["result"]

    async def toll_run(self, message: str, tool_name: str) -> Any:
        """Устаревшее имя; используйте ``tool_run``."""
        return await self.tool_run(message, tool_name)

    async def chat_loop(self, user_id: str, user_input: Optional[str] = None) -> str:
        if user_input is not None:
            await self.add_human_message(user_input, user_id)
        messages = self.conversation_history[user_id]
        ai_response, _tokens = await self._llm_chat_messages(messages)
        ai_response = await self.check_response(ai_response)
        await self.add_ai_message(ai_response, user_id)
        return ai_response

    @staticmethod
    async def check_response(ai_response: str) -> str:
        cleaned = ai_response.split("Пользователь:")[0]
        leak_markers = (
            "ТЕКУЩАЯ ЦЕЛЬ:",
            "ПРЕДЫДУЩИЙ ВОПРОС АССИСТЕНТА:",
            "ПРЕДЫДУЩИЙ ОТВЕТ ПОЛЬЗОВАТЕЛЯ:",
            "🧭",
        )
        for marker in leak_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        return cleaned.strip()

    async def start_talk(self, user_input: str, user_id: str, parameters: Optional[dict] = None) -> Any:
        parameters = parameters or {}
        self.user_last_seen[user_id] = datetime.now()
        await self._hydrate_from_crud(user_id)

        dn = parameters.get("user_display_name")
        if dn:
            self.user_metadata.setdefault(user_id, {})
            name = str(dn).strip()[:256]
            if name:
                self.user_metadata[user_id]["user_display_name"] = name

        if parameters.get("finish_test_early"):
            if self.user_state[user_id] != UserState.TEST:
                return "Сейчас тест не активен."

        current_input: Optional[str] = user_input

        while True:
            st = self.user_state[user_id]

            if st == UserState.WHO:
                if not self.conversation_history.get(user_id):
                    system_prompt = config["who_are_you_prompt"]
                    await self.add_system_message(system_prompt, user_id)
                ai_response = await self.chat_loop(user_id, current_input)
                new_state = await self.update_user_state(ai_response, user_id)
                if not new_state:
                    return ai_response
                self.conversation_history[user_id] = []
                current_input = None
                continue

            if st == UserState.ABOUT:
                if not self.conversation_history.get(user_id):
                    user_type = self.user_type[user_id]
                    if user_type == UserType.SCHOOL:
                        system_prompt = config["about_school_prompt"]
                    elif user_type == UserType.STUDENT:
                        system_prompt = config["about_student_prompt"]
                    else:
                        system_prompt = config["about_worker_prompt"]
                    system_prompt = system_prompt.replace("<user_metadata>", self.user_metadata[user_id]["who_user"])
                    await self.add_system_message(system_prompt, user_id)

                ai_response = await self.chat_loop(user_id, current_input)
                new_state = await self.update_user_state(ai_response, user_id)
                if not new_state:
                    return ai_response
                self.conversation_history[user_id] = []
                current_input = None
                continue

            if st == UserState.TEST:
                if self.test_variant == "v1" and self.skip_test_v1:
                    self.user_metadata[user_id]["test_user"] = (
                        "Этап тестирования v1 был пропущен по системной настройке."
                    )
                    self.set_user_state(
                        user_id,
                        UserState.RECOMMENDATION,
                        "skip_test_v1_to_recommendation_env_flag",
                    )
                    self.conversation_history[user_id] = []
                    current_input = None
                    continue
                if self.test_variant == "v2":
                    if not self.user_metadata[user_id].get("test_for_user"):
                        return await self._test_v2.recommend_entry(self, user_id)
                    out = await self._test_v2.handle_turn(self, user_id, current_input or "", parameters)
                    if out is not None:
                        return out
                    current_input = None
                    continue
                out = await self._test_v1.handle_turn(self, user_id, current_input, parameters)
                if out is not None:
                    return out
                current_input = None
                continue

            if st == UserState.RECOMMENDATION:
                system_prompt = config["recommend_profession_prompt"]
                await self.add_system_message(system_prompt, user_id)
                keys_with_info = ["who_user", "about_user", "test_user"]
                parts = []
                for key in keys_with_info:
                    if self.user_metadata[user_id].get(key) is not None:
                        parts.append(self.user_metadata[user_id][key])
                u_in = "\n".join(parts)
                u_in = f"\n\nИнформация о пользователе:\n{u_in}"
                ai_response = await self.chat_loop(user_id, u_in)
                self.user_metadata[user_id]["ai_recommendation"] = ai_response
                raw_json = await self.tool_run(ai_response, tool_name="make_json_tool")
                self.user_metadata[user_id]["ai_recommendation_json"] = _attach_profession_order(
                    normalize_ai_recommendation_json(raw_json)
                )
                self.set_user_state(user_id, UserState.TALK, "recommendation_phase_completed")
                self.conversation_history[user_id] = []
                return ai_response

            if st == UserState.TALK:
                system_prompt = config["talk_prompt"]
                system_prompt = system_prompt.replace("<who_user>", self.user_metadata[user_id]["who_user"])
                system_prompt = system_prompt.replace("<about_user>", self.user_metadata[user_id]["about_user"])
                system_prompt = system_prompt.replace("<test_user>", self.user_metadata[user_id]["test_user"])
                if self.user_metadata[user_id].get("ai_recommendation_json"):
                    system_prompt = system_prompt.replace(
                        "<ai_recommendation_json>",
                        str(self.user_metadata[user_id]["ai_recommendation_json"]["professions"]),
                    )
                else:
                    system_prompt = system_prompt.replace("# РЕКОМЕНДОВАННЫЕ ПРОФЕССИИ:", "")
                if not self.conversation_history[user_id]:
                    self.conversation_history[user_id].append({"role": "system", "text": system_prompt})
                elif self.conversation_history[user_id][0].get("role") == "system":
                    self.conversation_history[user_id][0]["text"] = system_prompt
                else:
                    self.conversation_history[user_id].insert(0, {"role": "system", "text": system_prompt})
                ai_response = await self.chat_loop(user_id, current_input)
                new_recommendation = await self.tool_run(ai_response, tool_name="is_recommendation_tool")
                if new_recommendation and new_recommendation.get("new_recommendation"):
                    extracted = normalize_ai_recommendation_json(
                        await self.tool_run(ai_response, tool_name="make_json_tool")
                    )
                    profs = (
                        normalize_professions(extracted.get("professions"))
                        if isinstance(extracted, dict)
                        else None
                    )
                    if profs:
                        self.user_metadata[user_id]["ai_recommendation"] = ai_response
                        self.user_metadata[user_id]["ai_recommendation_json"] = _attach_profession_order(extracted)
                return ai_response

            return None

    async def go_rag(self, profession_name: str, user_id: str) -> str:
        return await profession_rag.go_rag(self, profession_name, user_id)

    async def go_rag_roadmap(self, profession_name: str, user_id: str) -> str:
        return await profession_rag.go_rag_roadmap(self, profession_name, user_id)
