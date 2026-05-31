"""RAG по профессиям и роадмапы курсов (кэш контекста по имени профессии)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from common.rag_client import rag_search
from core.config_data import EDUCATION_JSON, api_key, config, folder_id

if TYPE_CHECKING:
    from core.dialog_model import DialogModel


class ProfessionRagService:
    """Кэш описаний/флагов контекста между запросами в рамках процесса core."""

    def __init__(self) -> None:
        self._prof_context: Dict[str, Dict[str, Any]] = {}

    def _web_search(self, dm: DialogModel):
        return dm._web_search_client()

    async def go_rag(self, dm: DialogModel, profession_name: str, user_id: str) -> str:
        profession_full = dm.user_metadata[user_id]["ai_recommendation_json"]["professions"][profession_name]
        profession_full = f"{profession_name}\n{profession_full}"
        prof_info = await rag_search(query=profession_full, k=2, api_key=api_key, folder_id=folder_id)
        about_profession = ""
        for doc in prof_info:
            about_profession += f"{doc[0].page_content}\n\n"

        profession = config["is_docs_about_profession_prompt"]
        profession = profession.replace("<text1>", profession_full)
        profession = profession.replace("<text2>", about_profession)

        is_context = None

        if profession_name in self._prof_context:
            profession_desc = self._prof_context[profession_name].get("description")
            if profession_desc:
                await dm.add_ai_message(profession_desc, user_id)
                return profession_desc
            is_context = self._prof_context[profession_name].get("is_context")
        else:
            self._prof_context[profession_name] = {}

        if is_context is None:
            docs_about_profession = await dm.tool_run(
                message=profession, tool_name="is_docs_about_profession_tool"
            )
            is_context = docs_about_profession["is_context"] if docs_about_profession else False
            self._prof_context[profession_name]["is_context"] = is_context

        if is_context:
            system_promt = config["describe_profession_prompt"]
            system_promt = system_promt.replace("<about_professions>", profession)
        else:
            system_promt = config["describe_profession_with_no_context_prompt"]

        system_promt = system_promt.replace("<profession>", profession_name)
        messages = [{"role": "system", "text": system_promt}]
        profession_desc, _t = await dm._llm_chat_messages(messages)
        self._prof_context[profession_name]["description"] = profession_desc
        await dm.add_ai_message(profession_desc, user_id)
        return profession_desc

    async def go_rag_roadmap(self, dm: DialogModel, profession_name: str, user_id: str) -> str:
        profession_desc = self._prof_context[profession_name]["description"]
        courses = await rag_search(
            query=profession_desc, k=15, api_key=api_key, folder_id=folder_id, index_dir="COURSES_DIR"
        )
        about_courses = ""
        with open(EDUCATION_JSON, encoding="utf-8") as f:
            links_data = f.read()
            links = json.loads(links_data)

        for doc in courses:
            about_courses += f"{doc[0].page_content}\n"
            link_name = doc[0].metadata["key"]
            link_data = links.get(link_name, None)
            if link_data:
                about_courses += f"Ссылка: {links[link_name]['link']}\n"
            about_courses += 80 * "_"
            about_courses += "\n\n"

        courses_match = f"{profession_name}\n{about_courses}"
        is_context = False
        docs_about_courses = await dm.tool_run(message=courses_match, tool_name="is_docs_about_courses_tool")
        if self._prof_context.get(profession_name):
            is_context = docs_about_courses["is_context"]

        courses_prompt = config["create_roadmap_prompt"]

        if not is_context:
            system = (
                "Сформируй поисковой запрос, по которому в интернете можно найти курсы для указанной"
                "профессии. Верни только поисковой запрос"
            )
            profession_full = dm.user_metadata[user_id]["ai_recommendation_json"]["professions"][profession_name]
            messages = [{"role": "system", "text": system}, {"role": "user", "text": f"{profession_full}"}]
            courses_from_web, _tok = await dm._llm_chat_messages(messages)
            try:
                about_courses = await self._web_search(dm).create_course_info(
                    query=courses_from_web, max_results=3
                )
                courses_prompt = courses_prompt.replace("<about_courses>", about_courses)
            except Exception:
                courses_prompt = config["create_roadmap_no_context_prompt"]

        courses_prompt = courses_prompt.replace("<about_profession>", profession_desc)
        courses_prompt = courses_prompt.replace("<profession>", profession_name)
        courses_prompt = courses_prompt.replace("<who_user>", dm.user_metadata[user_id]["who_user"])
        courses_prompt = courses_prompt.replace("<about_user>", dm.user_metadata[user_id]["about_user"])

        messages = [{"role": "system", "text": courses_prompt}]
        roadmap, _t = await dm._llm_chat_messages(messages)
        await dm.add_ai_message(roadmap, user_id)
        return roadmap


profession_rag = ProfessionRagService()
