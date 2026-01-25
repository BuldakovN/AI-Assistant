"""
Основной модуль Telegram бота
"""
import asyncio
import logging
import sys
import re
from typing import Dict, Any, Iterable
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove, KeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder

from config import config
from llm_client import LLMClient, LLMResponse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Основной класс Telegram бота"""
    
    def __init__(self):
        if not config.validate():
            raise ValueError("Некорректная конфигурация бота")
        
        self.bot = Bot(token=config.bot_token)
        self.dp = Dispatcher()
        self.user_sessions: Dict[int, Dict[str, Any]] = {}
        
        # Регистрируем обработчики
        self._register_handlers()
        # self.point_glued_to_newline_bug_re = re.compile('\\n[]')
        self.point_glued_to_newline_bug_reverse_re = re.compile('\*\\n')
        self.md_spec_symb_remove_re = re.compile('\*{2}|#+')
        self.cleanup_re = re.compile(r'#+\s|\*\*')
        self.link_deletus_re = re.compile(r'\[.+\](?=\()')
        # self.bold_re = re.compile('(?<=\-).+(?=\:)')
    
    def _register_handlers(self):
        """Регистрирует все обработчики команд и сообщений"""
        
        # Обработчик команды /start
        @self.dp.message(CommandStart())
        async def start_handler(message: Message):
            await self._handle_start(message)
        
        # Обработчик команды /help
        @self.dp.message(Command("help"))
        async def help_handler(message: Message):
            await self._handle_help(message)

        # Обработчик команды /clean_history
        @self.dp.message(Command("clean_history"))
        async def cancel_handler(message: Message):
            await self._handle_clean_history(message)
        
        # Обработчик команды /cancel
        # @self.dp.message(Command("cancel"))
        # async def cancel_handler(message: Message):
        #     await self._handle_cancel(message)
        
        # Обработчик всех остальных сообщений
        @self.dp.message()
        async def message_handler(message: Message):
            await self._handle_message(message)
        
        # Обработчик callback-кнопок для профессий
        @self.dp.callback_query()
        async def callback_handler(callback_query: CallbackQuery):
            await self._handle_callback(callback_query)
    
    async def _handle_start(self, message: Message):
        """Обработчик команды /start"""
        user_id = message.from_user.id
        
        # Инициализируем сессию пользователя
        self.user_sessions[user_id] = {
            "active": True,
            "message_count": 0
        }

        # Показываем, что бот печатает
        await self.bot.send_chat_action(user_id, "typing")
        
        await message.answer(config.start_text, parse_mode="Markdown")
        logger.info(f"Пользователь {user_id} запустил бота")
    
    async def _handle_help(self, message: Message):
        """Обработчик команды /help"""
        user_id = message.from_user.id

        # Показываем, что бот печатает
        await self.bot.send_chat_action(user_id, "typing")
        
        await message.answer(config.help_text)
        logger.info(f"Пользователь {user_id} запросил справку")

    async def _handle_clean_history(self, message: Message):
        """Обработчик команды /help"""
        user_id = message.from_user.id

        # Показываем, что бот печатает
        await self.bot.send_chat_action(user_id, "typing")
        async with LLMClient() as llm_client:
            response = await llm_client.clean_user_history(user_id)
            remove_keyboard = ReplyKeyboardRemove()

            # Сбрасываем процесс тестирования
            if self.user_sessions.get(user_id):
                self.user_sessions[user_id].pop("testing_process", None)
        await message.answer(response, parse_mode="Markdown", reply_markup=remove_keyboard)
        logger.info(f"Пользователь {user_id} запросил очистку истории")
    
    async def _handle_cancel(self, message: Message):
        """Обработчик команды /cancel"""
        user_id = message.from_user.id
        
        # Сбрасываем сессию пользователя
        if user_id in self.user_sessions:
            self.user_sessions[user_id]["active"] = False
        
        await message.answer(config.cancel_text, parse_mode="Markdown")
        logger.info(f"Пользователь {user_id} отменил операцию")

    async def _handle_message(self, message: Message):
        """Обработчик обычных сообщений"""
        user_id = message.from_user.id
        user_message = message.text

        # Проверяем активность сессии пользователя
        if user_id not in self.user_sessions or not self.user_sessions[user_id]["active"]:
            await message.answer("Пожалуйста, начните с команды /start", parse_mode="Markdown")
            return

        # Если пользователь в процессе тестирования и ждет ответа
        if (self.user_sessions[user_id].get("testing_process") and
                self.user_sessions[user_id]["testing_process"]["enabled"] and
                self.user_sessions[user_id]["testing_process"].get("awaiting_answer")):
            await self._process_test_answer(message)
            return

        # Если тестирование включено, но не ждем ответа (начало теста)
        if self.user_sessions[user_id].get("testing_process") and self.user_sessions[user_id]["testing_process"][
            "enabled"]:
            await self.run_test(message)

        # Обычная обработка сообщения
        self.user_sessions[user_id]["message_count"] += 1

        typing_task = asyncio.create_task(self._show_typing_indicator(user_id))

        try:
            async with LLMClient() as llm_client:
                llm_response = await llm_client.generate_response(user_message, user_id)

                if llm_response.test_info:
                    # Инициализируем процесс тестирования
                    self.user_sessions[user_id]["testing_process"] = {
                        "enabled": True,
                        "test_info": llm_response.test_info,
                        "answers": [],
                        "awaiting_answer": False
                    }
                    await self.run_test(message)
                else:
                    await self._send_response_with_professions(message, llm_response)
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            await message.answer("Извините, произошла ошибка при обработке вашего сообщения.", parse_mode="Markdown")
            typing_task.cancel()
            return

        logger.info(
            f"Отправлен ответ пользователю {user_id}, сообщение #{self.user_sessions[user_id]['message_count']}")
        typing_task.cancel()

    async def run_test(self, message: Message):
        user_id = message.from_user.id
        testing_process = self.user_sessions[user_id]["testing_process"]
        test_info = testing_process["test_info"]

        # Инициализируем состояние теста при первом запуске
        if "current_question_index" not in testing_process:
            testing_process["current_question_index"] = 0
            testing_process["answers"] = []
            testing_process["awaiting_answer"] = False
            # Отправляем описание теста
            test_desc = (f'Спасибо за ответы! Сейчас я проведу небольшой тест, чтобы на его основе подобрать профессии\n\n'
                         f'{test_info["test_description"]}')
            await message.answer(
                test_desc,
                parse_mode="Markdown"
            )

        current_idx = testing_process["current_question_index"]

        # Если мы ждем ответа на предыдущий вопрос, обрабатываем его
        if testing_process["awaiting_answer"]:
            await self._process_test_answer(message)
            return

        # Проверяем, есть ли еще вопросы
        if current_idx < len(test_info["test_questions"]):
            # Создаем клавиатуру с вариантами ответов
            buttons_list = test_info["test_bottoms"]
            builder = ReplyKeyboardBuilder()

            for test_button in buttons_list:
                builder.button(text=test_button)

            builder.row(KeyboardButton(text="🚫 Завершить тест"))
            keyboard = builder.as_markup(resize_keyboard=True)

            # Отправляем текущий вопрос
            await message.answer(
                test_info["test_questions"][current_idx],
                reply_markup=keyboard,
                parse_mode="Markdown"
            )

            # Устанавливаем флаг ожидания ответа
            testing_process["awaiting_answer"] = True

        else:
            # Все вопросы заданы, завершаем тест
            await self.finish_test(message)

    async def _process_test_answer(self, message: Message):
        user_id = message.from_user.id
        testing_process = self.user_sessions[user_id]["testing_process"]
        test_info = testing_process["test_info"]

        if message.text == "🚫 Завершить тест":
            await self.finish_test(message)
            return

        # Проверяем, что ответ является допустимым вариантом
        if message.text in test_info["test_bottoms"]:
            # Сохраняем ответ
            testing_process["answers"].append(message.text)

            # Увеличиваем счетчик вопроса
            testing_process["current_question_index"] += 1
            testing_process["awaiting_answer"] = False

            # Продолжаем со следующим вопросом или завершаем
            if testing_process["current_question_index"] < len(test_info["test_questions"]):
                await self.run_test(message)
            else:
                await self.finish_test(message)
        else:
            await message.answer("Пожалуйста, выберите один из предложенных вариантов")

    async def finish_test(self, message: Message):
        user_id = message.from_user.id
        testing_process = self.user_sessions[user_id]["testing_process"]

        # Собираем результаты
        answers = testing_process["answers"]

        # Убираем клавиатуру
        remove_keyboard = ReplyKeyboardRemove()
        await message.answer("Спасибо за прохождение теста! Подбираю подходящие профессии...", reply_markup=remove_keyboard)

        # Сбрасываем процесс тестирования
        questions = self.user_sessions[user_id]['testing_process']['test_info']['test_questions']
        self.user_sessions[user_id]["testing_process"] = {
            "enabled": False,
            "test_info": None
        }
        async with LLMClient() as llm_client:
            llm_response = await llm_client.generate_response('', user_id, parameters={"test_results": list(zip(questions, answers))})
            await self._send_response_with_professions(message, llm_response)



    async def _send_response_with_professions(self, original_message: Message, llm_response: LLMResponse):
        """
        Отправляет ответ от LLM с кнопками профессий, если они есть
        
        Args:
            original_message: Исходное сообщение пользователя
            llm_response: Ответ от LLM с текстом и профессиями
        """
        logger.info(f"🔍 Отправляем ответ с профессиями. Профессии: {llm_response.professions}")
        
        # Сначала отправляем основное сообщение
        await self._send_long_message(original_message, llm_response.msg)
        
        # Если есть профессии, создаем кнопки
        if llm_response.professions:
            logger.info(f"Создаем кнопки для {len(llm_response.professions)} профессий")
            builder = InlineKeyboardBuilder()
            
            user_id = original_message.from_user.id
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {"active": True, "message_count": 0}
            
            # Создаем мапинг профессий с индексами
            profession_list = list(llm_response.professions.items())
            self.user_sessions[user_id]["professions"] = profession_list
            
            for index, (profession_name, profession_description) in enumerate(profession_list):
                # Используем короткий индекс как callback_data
                builder.button(
                    text=profession_name,
                    callback_data=f"prof:{index}"
                )
                logger.info(f"➕ Добавлена кнопка: {profession_name} (индекс: {index})")
            
            # Располагаем кнопки по 2 в ряд
            builder.adjust(2)
            
            await original_message.answer(
                "💼 Выберите интересующую профессию для получения подробной информации:",
                reply_markup=builder.as_markup(),
                parse_mode="Markdown"
            )
            logger.info("✅ Кнопки отправлены")
        else:
            logger.info("ℹ️ Профессии отсутствуют, кнопки не создаем")
    
    async def _handle_callback(self, callback_query: CallbackQuery):
        """Обработчик callback-кнопок"""
        user_id = callback_query.from_user.id
        callback_data = callback_query.data
        
        # Подтверждаем получение callback
        await callback_query.answer()
        
        if callback_data == "back_to_professions":
            await self._show_professions_list(callback_query)
        else:
            try:
                # Извлекаем индекс профессии
                profession_index = int(callback_data.replace("prof:", "").replace("road:", ""))
                
                # Получаем профессию из сессии пользователя
                if (user_id in self.user_sessions and 
                    "professions" in self.user_sessions[user_id] and
                    0 <= profession_index < len(self.user_sessions[user_id]["professions"])):
                    
                    profession_name, profession_description = self.user_sessions[user_id]["professions"][profession_index]
                    if callback_data.startswith("prof:"):
                        await self._show_profession_details(callback_query, profession_name, profession_index)
                    elif callback_data.startswith("road:"):
                        await self._show_profession_roadmap(callback_query, profession_name)
                    else:
                        raise NotImplementedError
                else:
                    await callback_query.message.answer("❌ Профессия не найдена. Попробуйте еще раз.", parse_mode="Markdown")
                    logger.error(f"Профессия с индексом {profession_index} не найдена для пользователя {user_id}")
                    
            except (ValueError, IndexError, NotImplementedError) as e:
                await callback_query.message.answer("❌ Ошибка при обработке выбора профессии.", parse_mode="Markdown")
                logger.error(f"Ошибка обработки callback {callback_data}: {e}")
                if isinstance(e, NotImplementedError):
                    logger.error(f"Возникло неожиданное значение в callback кнопки: {callback_data}")
                    

    
    async def _show_profession_roadmap(self, callback_query: CallbackQuery, profession_name: str):
        """Показывает подробное описание профессии через RAG систему"""
        user_id = callback_query.from_user.id
        
        # Показываем, что бот обрабатывает запрос
        typing_task = asyncio.create_task(self._show_typing_indicator(user_id))

        # Создаем кнопку "Назад к списку профессий" даже при ошибке
        builder = InlineKeyboardBuilder()
        builder.button(
            text="🔙 Назад к списку профессий",
            callback_data="back_to_professions"
        )

        try:
            # Запрашиваем у LLM подробное описание профессии через RAG
            async with LLMClient() as llm_client:
                llm_response = await llm_client.get_profession_roadmap(profession_name, user_id)

                # Отправляем описание профессии с кнопкой "Назад"
                await self._send_long_message(callback_query.message, llm_response.msg, builder)

        except Exception as e:
            typing_task.cancel()
            logger.error(f"Ошибка при получении описания профессии {profession_name}: {e}")
            
            await callback_query.message.answer(
                f"📋 **{profession_name}**\n\nИзвините, не удалось получить подробное описание этой профессии.",
                reply_markup=builder.as_markup(),
                parse_mode="Markdown"
            )
        
        logger.info(f"Пользователь {user_id} запросил описание профессии: {profession_name}")
        typing_task.cancel()

    async def _show_profession_details(self, callback_query: CallbackQuery, profession_name: str, prof_index: int):
        """Показывает подробное описание профессии через RAG систему"""
        user_id = callback_query.from_user.id
        
        # Показываем, что бот обрабатывает запрос
        typing_task = asyncio.create_task(self._show_typing_indicator(user_id))

        # Создаем кнопку "Назад к списку профессий" даже при ошибке
        builder = InlineKeyboardBuilder()
        builder.button(
            text="🔙 Назад к списку профессий",
            callback_data="back_to_professions"
        )

        try:
            # Запрашиваем у LLM подробное описание профессии через RAG
            async with LLMClient() as llm_client:
                llm_response = await llm_client.get_profession_info(profession_name, user_id)
                builder.button(
                    text="📈 Показать роудмап",
                    callback_data=f"road:{prof_index}"
                )
                # Отправляем описание профессии с кнопкой "Назад"
                await self._send_long_message(callback_query.message, llm_response.msg, builder)

        except Exception as e:
            typing_task.cancel()
            logger.error(f"Ошибка при получении описания профессии {profession_name}: {e}")
            
            await callback_query.message.answer(
                f"📋 **{profession_name}**\n\nИзвините, не удалось получить подробное описание этой профессии.",
                reply_markup=builder.as_markup(),
                parse_mode="Markdown"
            )
        
        logger.info(f"Пользователь {user_id} запросил описание профессии: {profession_name}")
        typing_task.cancel()
    
    async def _show_professions_list(self, callback_query: CallbackQuery):
        """Показывает список профессий с кнопками"""
        user_id = callback_query.from_user.id
        
        # Проверяем, есть ли сохраненные профессии в сессии пользователя
        if (user_id in self.user_sessions and 
            "professions" in self.user_sessions[user_id] and
            self.user_sessions[user_id]["professions"]):
            
            professions = self.user_sessions[user_id]["professions"]
            builder = InlineKeyboardBuilder()
            
            # Создаем кнопки для всех профессий
            for index, (profession_name, profession_description) in enumerate(professions):
                builder.button(
                    text=profession_name,
                    callback_data=f"prof:{index}"
                )
            
            # Располагаем кнопки по 2 в ряд
            builder.adjust(2)
            
            # Редактируем сообщение с кнопкой "Назад", убирая её и показывая список профессий
            try:
                await callback_query.message.edit_text(
                    "💼 Выберите интересующую профессию для получения подробной информации:",
                    reply_markup=builder.as_markup(),
                    parse_mode="Markdown"
                )
                logger.info(f"Пользователь {user_id} вернулся к списку профессий")
            except Exception as e:
                # Если редактирование не удалось, отправляем новое сообщение
                logger.warning(f"Не удалось отредактировать сообщение: {e}")
                await callback_query.message.answer(
                    "💼 Выберите интересующую профессию для получения подробной информации:",
                    reply_markup=builder.as_markup(),
                    parse_mode="Markdown"
                )
                logger.info(f"Пользователь {user_id} вернулся к списку профессий (новое сообщение)")
        else:
            # Если профессии не найдены, отправляем сообщение об ошибке
            await callback_query.message.answer(
                "❌ Список профессий не найден. Попробуйте начать новый диалог с помощью команды /start",
                parse_mode="Markdown"
            )
            logger.warning(f"Профессии не найдены для пользователя {user_id}")

    def sanitize_text(self, text):
        text = self.cleanup_re.sub('', text)
        text = text.replace('*', '-')
        text = self.link_deletus_re.sub('', text)
        return text

    async def _send_long_message(self, original_message: Message, text: str, buttons: InlineKeyboardBuilder = None):
        """
        Отправляет длинное сообщение, разбивая его на части если необходимо

        Args:
            original_message: Исходное сообщение пользователя
            text: Текст для отправки
        """
        text = self.sanitize_text(text)
        # if len(text) <= config.max_message_length:
        #     if buttons:
        #         await original_message.answer(text, parse_mode="Markdown", reply_markup=buttons.as_markup())
        #     else:
        #         await original_message.answer(text, parse_mode="Markdown")
        #     return

        # Просто разбиваем текст на куски фиксированной длины
        parts = []
        for i in range(0, len(text), config.max_message_length):
            part = text[i:i + config.max_message_length]
            parts.append(part)

        # Отправляем все части
        for i, part in enumerate(parts):
            if i > 0:
                await asyncio.sleep(config.message_delay)
            # прикрепляем кнопку к последнему сообщению
            if i == len(parts) - 1 and buttons:
                await original_message.answer(part, reply_markup=buttons.as_markup())
            else:
                await original_message.answer(part)

    async def _show_typing_indicator(self, user_id):
        """Показывать индикатор набора каждые 3 секунды пока не отменят"""
        while True:
            try:
                await self.bot.send_chat_action(user_id, "typing")
                await asyncio.sleep(3)  # Telegram показывает 5 секунд
            except asyncio.CancelledError:
                break
            except Exception:
                break
    
    async def start_polling(self):
        """Запускает бота в режиме polling"""
        logger.info("Запуск бота...")
        try:
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}")
            raise
    
    async def stop(self):
        """Останавливает бота"""
        logger.info("Остановка бота...")
        await self.bot.session.close()

# Функция для запуска бота
async def main():
    """Основная функция для запуска бота"""
    bot = TelegramBot()
    
    try:
        await bot.start_polling()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
