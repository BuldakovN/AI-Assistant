"""Пошаговый тест (v2): test_session, кнопки, досрочное завершение."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.config_data import config, prof_tests
from core.constants import UserState
from core.test_utils import canonical_prof_test_key, convert_to_qa_format

if TYPE_CHECKING:
    from core.dialog_model import DialogModel


class TestFlowV2:
    def prof_test_description(self, test_for_user: str) -> str:
        desc_map = config.get("prof_tests_description") or {}
        if isinstance(desc_map, dict) and test_for_user in desc_map:
            return str(desc_map[test_for_user])
        return str((prof_tests.get("test_description") or {}).get(test_for_user, ""))

    async def finalize_to_recommendation(self, dm: DialogModel, user_id: str, collected: List) -> None:
        meta = dm.user_metadata[user_id]
        test_for_user = meta.get("test_for_user")
        if not test_for_user:
            meta.pop("test_session", None)
            dm.set_user_state(user_id, UserState.RECOMMENDATION, "test_v2_finalize_no_test_for_user")
            dm.conversation_history[user_id] = []
            return

        test_description = self.prof_test_description(test_for_user)
        rules = str((prof_tests.get("test_description") or {}).get(test_for_user, ""))

        if len(collected) > 2:
            user_answers = convert_to_qa_format(collected)
            system_prompt = (
                f"Перед тобой результаты тестирования пользователя. Был проведен {test_for_user}\n\n"
                f"ОПИСАНИЕ ТЕСТА:\n{test_description}\n\nПроанализируй ответы пользователя."
                f"Сделай суммаризацию информации, выдели ключевые аспекты"
            )
            user_prompt = f"ПРАВИЛА ПРОХОЖДЕНИЯ ТЕСТА:\n{rules}\n\nТЕСТИРОВАНИЕ ПОЛЬЗОВАТЕЛЯ:\n{user_answers}"
            await dm.add_system_message(system_prompt, user_id)
            ai_response = await dm.chat_loop(user_id, user_prompt)
            meta["test_user"] = ai_response
        elif len(collected) > 0:
            meta["test_user"] = convert_to_qa_format(collected)
        else:
            meta["test_user"] = "Тест завершён досрочно: ответы на вопросы не были получены."

        meta.pop("test_session", None)
        dm.set_user_state(user_id, UserState.RECOMMENDATION, "test_v2_finalize")
        dm.conversation_history[user_id] = []

    async def handle_turn(
        self,
        dm: DialogModel,
        user_id: str,
        user_input: str,
        parameters: Dict[str, Any],
    ) -> Optional[str]:
        meta = dm.user_metadata[user_id]
        rt = meta.get("recommended_test") or {}
        questions: List[str] = list(rt.get("test_questions") or [])
        bottoms: List[str] = list(rt.get("test_bottoms") or [])

        tr_legacy = parameters.get("test_results")
        if tr_legacy:
            pairs = [list(x) for x in tr_legacy]
            meta["test_session"] = {"current_index": len(questions), "answers": pairs}
            await self.finalize_to_recommendation(dm, user_id, pairs)
            return None

        if parameters.get("finish_test_early"):
            ts = meta.get("test_session") or {"current_index": 0, "answers": []}
            collected = list(ts.get("answers") or [])
            await self.finalize_to_recommendation(dm, user_id, collected)
            return None

        ts = meta.get("test_session")
        if ts is None:
            ts = {"current_index": 0, "answers": []}
            meta["test_session"] = ts

        idx = int(ts.get("current_index") or 0)
        answers: List = list(ts.get("answers") or [])
        ts["answers"] = answers

        prompt = (user_input or "").strip()
        finish_markers = ("🚫 Завершить тест", "/finish_test")
        if prompt in finish_markers:
            await self.finalize_to_recommendation(dm, user_id, answers)
            return None

        if not questions:
            return "Не удалось загрузить вопросы теста. Попробуйте позже."

        if idx >= len(questions):
            await self.finalize_to_recommendation(dm, user_id, answers)
            return None

        if idx == 0 and len(answers) == 0 and not prompt:
            td = rt.get("test_description") or self.prof_test_description(meta["test_for_user"])
            return (
                "Спасибо за ответы! Сейчас я проведу небольшой тест, чтобы на его основе подобрать профессии\n\n"
                f"{td}\n\n{questions[0]}"
            )

        if not prompt:
            return questions[idx]

        if prompt not in bottoms:
            return (
                "Пожалуйста, выберите один из предложенных вариантов ответа.\n\n"
                f"{questions[idx]}"
            )

        answers.append([questions[idx], prompt])
        idx += 1
        ts["current_index"] = idx
        ts["answers"] = answers

        if idx >= len(questions):
            await self.finalize_to_recommendation(dm, user_id, answers)
            return None

        return questions[idx]

    async def recommend_entry(self, dm: DialogModel, user_id: str) -> str:
        about_user = "\n".join(str(v) for v in dm.user_metadata[user_id].values())
        select_test_message = config["prof_test_description_for_tool"]
        select_test_message = select_test_message.replace("<about_user>", about_user)
        tool_out = await dm.tool_run(select_test_message, tool_name="select_test_tool")
        raw_name = tool_out["user_test"] if isinstance(tool_out, dict) else ""
        test_for_user = canonical_prof_test_key(prof_tests, raw_name)
        if test_for_user not in (prof_tests.get("test_questions") or {}):
            return (
                "Не удалось сопоставить выбранный тест с каталогом методик. "
                "Попробуйте отправить сообщение ещё раз или начните с /start."
            )
        dm.user_metadata[user_id]["test_for_user"] = test_for_user

        test_description = prof_tests["test_description"][test_for_user]
        test_questions = prof_tests["test_questions"][test_for_user]
        test_bottoms = prof_tests["test_bottoms"][test_for_user]
        dm.user_metadata[user_id]["recommended_test"] = {
            "test_description": test_description,
            "test_questions": test_questions,
            "test_bottoms": test_bottoms,
        }
        dm.user_metadata[user_id]["test_session"] = {"current_index": 0, "answers": []}
        td = self.prof_test_description(test_for_user)
        if not test_questions:
            return "Не удалось подобрать вопросы для теста. Попробуйте позже."
        q0 = test_questions[0]
        return (
            "Спасибо за ответы! Сейчас я проведу небольшой тест, чтобы на его основе подобрать профессии\n\n"
            f"{td}\n\n{q0}"
        )
