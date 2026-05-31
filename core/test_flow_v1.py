"""Тест v1: один чат-прогон с [EXIT]; досрочное завершение через ``finish_test_early`` (как /finish_test)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from core.config_data import config
from core.constants import UserState

if TYPE_CHECKING:
    from core.dialog_model import DialogModel


class TestFlowV1:
    async def recommend_test(self, dm: DialogModel, user_id: str) -> None:
        recommendation_prompt = config["test_recommendation_prompt"]
        await dm.add_system_message(recommendation_prompt, user_id)
        user_input = "\n".join(str(v) for v in dm.user_metadata[user_id].values())
        ai_response = await dm.chat_loop(user_id, user_input)
        dm.user_metadata[user_id]["recommended_test"] = ai_response
        dm.conversation_history[user_id] = []

    async def finish_early(self, dm: DialogModel, user_id: str) -> None:
        """Досрочный выход из теста v1: суммаризация переписки теста → recommendation (как при [EXIT])."""
        history = dm.conversation_history.get(user_id) or []
        has_user_turns = any((m.get("role") == "user") for m in history)
        if has_user_turns:
            user_story = await dm.summarization(user_id)
            dm.user_metadata[user_id]["test_user"] = user_story
        else:
            dm.user_metadata[user_id]["test_user"] = (
                "Тест завершён досрочно: ответы на вопросы теста не были получены."
            )
        dm.set_user_state(user_id, UserState.RECOMMENDATION, "test_v1_finish_early")
        dm.conversation_history[user_id] = []

    async def handle_turn(
        self,
        dm: DialogModel,
        user_id: str,
        user_input: Optional[str],
        parameters: Dict[str, Any],
    ) -> Optional[str]:
        if parameters.get("finish_test_early"):
            await self.finish_early(dm, user_id)
            return None

        if not dm.conversation_history.get(user_id):
            await self.recommend_test(dm, user_id)
            system_prompt = config["test_run_prompt"]
            user_metadata = (
                f"{dm.user_metadata[user_id]['who_user']}\n{dm.user_metadata[user_id]['about_user']}"
            )
            test = dm.user_metadata[user_id]["recommended_test"]
            system_prompt = system_prompt.replace("<user_metadata>", user_metadata)
            system_prompt = system_prompt.replace("<test>", test)
            await dm.add_system_message(system_prompt, user_id)
            user_input = "/start"

        ai_response = await dm.chat_loop(user_id, user_input)
        new_state = await dm.update_user_state(ai_response, user_id)
        if not new_state:
            return ai_response
        dm.conversation_history[user_id] = []
        return None
