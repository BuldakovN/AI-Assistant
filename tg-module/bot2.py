"""
Telegram-бот: только приём/отправка сообщений. Состояние диалога — в CRUD и ответах core.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, KeyboardButton, Message, ReplyKeyboardRemove
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram.client.session.aiohttp import AiohttpSession

from common.error_logging import setup_service_error_logging
from config import config
from llm_client import LLMClient, LLMResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
setup_service_error_logging("tg-module")


STATE_EMOJI = {
    "who": "👋",
    "about": "🧭",
    "test": "📝",
    "recommendation": "🎯",
    "talk": "💬",
    "inject_attempt": "🛡️",
}

FINISH_TEST_BUTTON = "🚫 Завершить тест"
PROF_LIST_TEXT = "💼 Выберите интересующую профессию для получения подробной информации:"
PROF_PREPARE_HINT = "⏳ Дайте пару секунд — готовлю подробную информацию."


def professions_from_crud_payload(payload: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Порядок имён: ``profession_order`` из core, иначе порядок ключей в ``professions``."""
    if not payload:
        return []
    meta = payload.get("user_metadata") or {}
    rec = meta.get("ai_recommendation_json") or {}
    prof = rec.get("professions")
    if not isinstance(prof, dict) or not prof:
        return []
    order = rec.get("profession_order")
    if isinstance(order, list) and order:
        keys: List[str] = []
        seen: set[str] = set()
        for k in order:
            if isinstance(k, str) and k in prof and k not in seen:
                keys.append(k)
                seen.add(k)
        for k in prof:
            if k not in seen:
                keys.append(k)
    else:
        keys = list(prof.keys())
    return [(k, prof[k]) for k in keys]


def crud_has_meaningful_session(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    user_state = payload.get("user_state") or "who"
    history = payload.get("conversation_history") or []
    metadata = payload.get("user_metadata") or {}
    return bool(
        payload.get("app_user_id") is not None
        or history
        or metadata
        or payload.get("user_type")
        or user_state != "who"
    )


class SafeMessageSender:
    """Markdown parse fallback; разбиение длинных сообщений."""

    def __init__(self, max_len: int, delay: float) -> None:
        self.max_len = max_len
        self.delay = delay

    async def send(
        self,
        message: Message,
        text: str,
        prefix: str,
        parse_mode: Optional[str] = "Markdown",
        reply_markup: Any = None,
    ) -> None:
        content = f"{prefix} {text}".strip()
        if len(content) <= self.max_len:
            await self._send_single(message, content, parse_mode=parse_mode, reply_markup=reply_markup)
            return

        chunks = self._split_markdown(content, self.max_len)
        for index, chunk in enumerate(chunks):
            if index > 0:
                await asyncio.sleep(self.delay)
            last_markup = reply_markup if index == len(chunks) - 1 else None
            await self._send_single(message, chunk, parse_mode=parse_mode, reply_markup=last_markup)

    async def _send_single(
        self,
        message: Message,
        text: str,
        parse_mode: Optional[str],
        reply_markup: Any,
    ) -> None:
        try:
            await message.answer(text, parse_mode=parse_mode, reply_markup=reply_markup)
        except TelegramBadRequest as exc:
            if "can't parse entities" not in str(exc).lower():
                raise
            await message.answer(text, reply_markup=reply_markup)

    def _split_markdown(self, text: str, limit: int) -> List[str]:
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        remaining = text

        while len(remaining) > limit:
            split_at = self._best_split_position(remaining, limit)
            if split_at <= 0:
                split_at = limit
            chunk = remaining[:split_at].rstrip()
            if not chunk:
                chunk = remaining[:limit]
                split_at = limit
            chunks.append(chunk)
            remaining = remaining[split_at:].lstrip()

        if remaining:
            chunks.append(remaining)
        return chunks

    def _best_split_position(self, text: str, limit: int) -> int:
        candidates = [m.start() for m in re.finditer(r"\n\n|\n|\. |\! |\? |, | ", text[:limit])]
        if not candidates:
            return limit

        for pos in reversed(candidates):
            left = text[:pos]
            if self._is_markdown_balanced(left):
                return pos
        return candidates[-1]

    @staticmethod
    def _is_markdown_balanced(text: str) -> bool:
        star = text.count("*") % 2 == 0
        under = text.count("_") % 2 == 0
        backtick = text.count("`") % 2 == 0
        square = text.count("[") == text.count("]")
        paren = text.count("(") == text.count(")")
        return star and under and backtick and square and paren


class TelegramBot:
    def __init__(self) -> None:
        if not config.validate():
            raise ValueError("Некорректная конфигурация бота")
        proxy = config.proxy
        session = AiohttpSession(proxy=proxy)
        self.bot = Bot(token=config.bot_token, session=session)
        self.dp = Dispatcher()
        self.sender = SafeMessageSender(config.max_message_length, config.message_delay)
        self.cleanup_re = re.compile(r"#+\s|\*\*")
        self.link_deletus_re = re.compile(r"\[.+\](?=\()")
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.dp.message(CommandStart())
        async def start_handler(message: Message) -> None:
            await self._handle_start(message)

        @self.dp.message(Command("help"))
        async def help_handler(message: Message) -> None:
            await self._handle_help(message)

        @self.dp.message(Command("clean_history"))
        async def clean_handler(message: Message) -> None:
            await self._handle_clean_history(message)

        @self.dp.message(Command(commands=["status", "stasut"]))
        async def status_handler(message: Message) -> None:
            await self._handle_status(message)

        @self.dp.message(Command("status_local"))
        async def status_local_handler(message: Message) -> None:
            await self._handle_status_local(message)

        @self.dp.message(Command("finish_test"))
        async def finish_handler(message: Message) -> None:
            await self._handle_finish_test_command(message)

        @self.dp.message()
        async def message_handler(message: Message) -> None:
            await self._handle_message(message)

        @self.dp.callback_query()
        async def callback_handler(callback_query: CallbackQuery) -> None:
            await self._handle_callback(callback_query)

    @staticmethod
    def _request_parameters(message: Message, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if message.from_user:
            name = (message.from_user.full_name or message.from_user.first_name or "").strip()
            if name:
                result["user_display_name"] = name
        if extra:
            result.update(extra)
        return result

    @staticmethod
    def _state_to_emoji(state: Optional[str]) -> str:
        return STATE_EMOJI.get((state or "").strip().lower(), "❓")

    async def _user_state_for_prefix(self, message: Message) -> str:
        uid = message.from_user.id if message.from_user else 0
        async with LLMClient() as llm:
            session = await llm.get_user_session(uid)
        return str((session or {}).get("user_state") or "who")

    async def _send_text(
        self,
        message: Message,
        text: str,
        parse_mode: Optional[str] = "Markdown",
        reply_markup: Any = None,
        user_state: Optional[str] = None,
    ) -> None:
        if user_state is None:
            user_state = await self._user_state_for_prefix(message)
        prefix = self._state_to_emoji(user_state)
        await self.sender.send(
            message=message,
            text=self.sanitize_text(text),
            prefix=prefix,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )

    def _reply_markup_for_llm(self, resp: LLMResponse) -> Any:
        state = (resp.user_state or "").strip().lower()
        info = resp.test_info if isinstance(resp.test_info, dict) else {}
        bottoms = info.get("test_bottoms")
        if state == "test" or bottoms:
            kb = ReplyKeyboardBuilder()
            if bottoms:
                for item in bottoms:
                    kb.button(text=str(item))
            kb.row(KeyboardButton(text=FINISH_TEST_BUTTON))
            placeholder = (
                "Выберите вариант ответа или завершите тест"
                if bottoms
                else "Ответьте текстом или нажмите «Завершить тест»"
            )
            return kb.as_markup(
                resize_keyboard=True,
                one_time_keyboard=False,
                is_persistent=True,
                input_field_placeholder=placeholder,
            )
        return ReplyKeyboardRemove()

    async def _dispatch_llm_response(self, message: Message, resp: LLMResponse) -> None:
        us = resp.user_state
        markup = self._reply_markup_for_llm(resp)
        if (resp.msg or "").strip():
            await self._send_text(message, resp.msg, reply_markup=markup, user_state=us)
        if resp.professions:
            if not (resp.msg or "").strip():
                await self._send_text(
                    message, "\u2060", reply_markup=ReplyKeyboardRemove(), user_state=us
                )
            profession_list = list(resp.professions.items())
            kb = InlineKeyboardBuilder()
            for idx, (name, _) in enumerate(profession_list):
                kb.button(text=name, callback_data=f"prof:{idx}")
            kb.adjust(2)
            await self._send_text(message, PROF_LIST_TEXT, reply_markup=kb.as_markup(), user_state=us)

    async def _show_typing_indicator(self, user_id: int) -> None:
        while True:
            try:
                await self.bot.send_chat_action(user_id, "typing")
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def sanitize_text(self, text: str) -> str:
        cleaned = self.cleanup_re.sub("", text or "")
        cleaned = cleaned.replace("*", "-")
        cleaned = self.link_deletus_re.sub("", cleaned)
        return cleaned

    async def _handle_start(self, message: Message) -> None:
        user_id = message.from_user.id
        async with LLMClient() as llm:
            resp = await llm.generate_response("/start", user_id, parameters=self._request_parameters(message))
        await self._dispatch_llm_response(message, resp)

    async def _handle_help(self, message: Message) -> None:
        await self.bot.send_chat_action(message.from_user.id, "typing")
        await self._send_text(message, config.help_text)

    async def _handle_clean_history(self, message: Message) -> None:
        user_id = message.from_user.id
        await self.bot.send_chat_action(user_id, "typing")
        async with LLMClient() as llm:
            text = await llm.clean_user_history(user_id)
        await self._send_text(message, text, reply_markup=ReplyKeyboardRemove(), user_state="who")

    async def _handle_status(self, message: Message) -> None:
        user_id = message.from_user.id
        await self.bot.send_chat_action(user_id, "typing")
        async with LLMClient() as llm:
            payload = await llm.get_user_session(user_id)
        if not payload:
            await self._send_text(message, "Не удалось получить статус. Попробуйте повторить чуть позже.")
            return

        state = payload.get("user_state") or "who"
        status_text = (
            "Текущий статус диалога (источник — CRUD):\n\n"
            f"- Фаза: `{state}`\n"
            f"- Тип пользователя: `{payload.get('user_type') or 'не определен'}`\n"
            f"- Сообщений в истории: `{len(payload.get('conversation_history') or [])}`\n"
            f"- Поля метаданных: `{', '.join(sorted((payload.get('user_metadata') or {}).keys())) or 'нет'}`\n"
            f"- Локальный кэш сессии в боте: не используется (stateless).\n"
        )
        await self._send_text(message, status_text, user_state=state)

    async def _handle_status_local(self, message: Message) -> None:
        user_id = message.from_user.id
        await self.bot.send_chat_action(user_id, "typing")
        async with LLMClient() as llm:
            crud = await llm.get_user_session(user_id)
        if not crud:
            await self._send_text(message, "Не удалось прочитать сессию из CRUD.")
            return

        meta = crud.get("user_metadata") or {}
        ts = meta.get("test_session")
        ts_line = "нет"
        if isinstance(ts, dict):
            ts_line = f"index=`{ts.get('current_index')}`, ответов=`{len(ts.get('answers') or [])}`"
        po = (meta.get("ai_recommendation_json") or {}).get("profession_order")
        po_line = ", ".join(str(x) for x in po) if isinstance(po, list) else "нет"
        snap = (
            "Снимок сессии (только CRUD; бот не хранит состояние в памяти):\n\n"
            f"- user_state: `{crud.get('user_state') or 'who'}`\n"
            f"- test_session: `{ts_line}`\n"
            f"- profession_order в рекомендации: `{po_line}`\n"
        )
        await self._send_text(message, snap, user_state=str(crud.get("user_state") or "who"))

    async def _handle_finish_test_command(self, message: Message) -> None:
        user_id = message.from_user.id
        await self.bot.send_chat_action(user_id, "typing")
        async with LLMClient() as llm:
            resp = await llm.finish_test_early(user_id, parameters=self._request_parameters(message))
        await self._dispatch_llm_response(message, resp)

    async def _handle_message(self, message: Message) -> None:
        user_id = message.from_user.id
        text = message.text or ""

        async with LLMClient() as llm:
            payload = await llm.get_user_session(user_id)
        if not crud_has_meaningful_session(payload):
            await self._send_text(message, "Пожалуйста, начните с команды /start")
            return

        if text == FINISH_TEST_BUTTON:
            await self.bot.send_chat_action(user_id, "typing")
            async with LLMClient() as llm:
                resp = await llm.finish_test_early(user_id, parameters=self._request_parameters(message))
            await self._dispatch_llm_response(message, resp)
            return

        typing_task = asyncio.create_task(self._show_typing_indicator(user_id))
        try:
            async with LLMClient() as llm:
                resp = await llm.generate_response(
                    text,
                    user_id,
                    parameters=self._request_parameters(message),
                )
            await self._dispatch_llm_response(message, resp)
        except Exception:
            logger.exception("Ошибка при генерации ответа")
            await self._send_text(message, "Извините, произошла ошибка при обработке вашего сообщения.")
        finally:
            typing_task.cancel()

    async def _handle_callback(self, callback_query: CallbackQuery) -> None:
        await callback_query.answer()
        user_id = callback_query.from_user.id
        data = callback_query.data or ""

        if data == "back_to_professions":
            await self._show_professions_list(callback_query)
            return

        if not callback_query.message:
            return

        try:
            index = int(data.replace("prof:", "").replace("road:", ""))
        except ValueError:
            await self._send_text(callback_query.message, "❌ Ошибка при обработке выбора профессии.")
            return

        async with LLMClient() as llm:
            payload = await llm.get_user_session(user_id)
        professions = professions_from_crud_payload(payload)
        if not (0 <= index < len(professions)):
            await self._send_text(callback_query.message, "❌ Профессия не найдена. Попробуйте еще раз.")
            return

        profession_name, _ = professions[index]
        prof_state = str((payload or {}).get("user_state") or "recommendation")
        if data.startswith("prof:"):
            await self._show_profession_details(callback_query, profession_name, index, user_state=prof_state)
        elif data.startswith("road:"):
            await self._show_profession_roadmap(callback_query, profession_name, user_state=prof_state)
        else:
            await self._send_text(callback_query.message, "❌ Ошибка при обработке выбора профессии.")

    async def _show_profession_details(
        self,
        callback_query: CallbackQuery,
        profession_name: str,
        index: int,
        *,
        user_state: Optional[str] = None,
    ) -> None:
        if not callback_query.message:
            return
        us = user_state or await self._user_state_for_prefix(callback_query.message)
        await self._send_text(callback_query.message, PROF_PREPARE_HINT, user_state=us)
        typing_task = asyncio.create_task(self._show_typing_indicator(callback_query.from_user.id))
        kb = InlineKeyboardBuilder()
        kb.button(text="🔙 Назад к списку профессий", callback_data="back_to_professions")
        kb.button(text="📈 Показать роудмап", callback_data=f"road:{index}")
        kb.adjust(1)
        try:
            async with LLMClient() as llm:
                resp = await llm.get_profession_info(profession_name, callback_query.from_user.id)
            await self._send_text(
                callback_query.message, resp.msg, reply_markup=kb.as_markup(), user_state=resp.user_state or us
            )
        except Exception:
            logger.exception("Ошибка при получении описания профессии %s", profession_name)
            await self._send_text(
                callback_query.message,
                f"📋 **{profession_name}**\n\nИзвините, не удалось получить подробное описание этой профессии.",
                reply_markup=kb.as_markup(),
                user_state=us,
            )
        finally:
            typing_task.cancel()

    async def _show_profession_roadmap(
        self,
        callback_query: CallbackQuery,
        profession_name: str,
        *,
        user_state: Optional[str] = None,
    ) -> None:
        if not callback_query.message:
            return
        us = user_state or await self._user_state_for_prefix(callback_query.message)
        await self._send_text(callback_query.message, PROF_PREPARE_HINT, user_state=us)
        typing_task = asyncio.create_task(self._show_typing_indicator(callback_query.from_user.id))
        kb = InlineKeyboardBuilder()
        kb.button(text="🔙 Назад к списку профессий", callback_data="back_to_professions")
        kb.adjust(1)
        try:
            async with LLMClient() as llm:
                resp = await llm.get_profession_roadmap(profession_name, callback_query.from_user.id)
            await self._send_text(
                callback_query.message, resp.msg, reply_markup=kb.as_markup(), user_state=resp.user_state or us
            )
        except Exception:
            logger.exception("Ошибка при получении roadmap профессии %s", profession_name)
            await self._send_text(
                callback_query.message,
                f"📋 **{profession_name}**\n\nИзвините, не удалось получить подробное описание этой профессии.",
                reply_markup=kb.as_markup(),
                user_state=us,
            )
        finally:
            typing_task.cancel()

    async def _show_professions_list(self, callback_query: CallbackQuery) -> None:
        if not callback_query.message:
            return
        async with LLMClient() as llm:
            payload = await llm.get_user_session(callback_query.from_user.id)
        professions = professions_from_crud_payload(payload)
        if not professions:
            await self._send_text(
                callback_query.message,
                "❌ Список профессий не найден. Попробуйте начать новый диалог с помощью команды /start",
            )
            return

        kb = InlineKeyboardBuilder()
        for idx, (name, _) in enumerate(professions):
            kb.button(text=name, callback_data=f"prof:{idx}")
        kb.adjust(2)

        state = str((payload or {}).get("user_state") or "who")
        prefixed = f"{self._state_to_emoji(state)} {PROF_LIST_TEXT}"
        try:
            await callback_query.message.edit_text(prefixed, parse_mode="Markdown", reply_markup=kb.as_markup())
        except TelegramBadRequest:
            await self._send_text(
                callback_query.message, PROF_LIST_TEXT, reply_markup=kb.as_markup(), user_state=state
            )

    async def start_polling(self) -> None:
        logger.info("Запуск бота...")
        await self.dp.start_polling(self.bot)

    async def stop(self) -> None:
        logger.info("Остановка бота...")
        await self.bot.session.close()


async def main() -> None:
    bot = TelegramBot()
    try:
        await bot.start_polling()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
