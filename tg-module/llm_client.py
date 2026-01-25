"""
Простая прослойка для вызова LLM API
"""
import aiohttp
from typing import Optional, Dict, Any
import os
import sys
from datetime import datetime

class LLMResponse:
    """Класс для представления ответа от LLM"""
    def __init__(self, msg: str, professions: Optional[Dict[str, str]] = None, test_info: Optional[Dict] = None):
        self.msg = msg
        self.professions = professions or {}
        self.test_info = test_info or {}


class LLMClient:
    """Простая прослойка - просто вызывает API из папки model"""
    
    def __init__(self):
        # API уже запущен на localhost:8000 с токенами
        self.api_url = os.getenv('LLM_API_URL', 'http://localhost:8000/start_talk/')
        self.profession_api_url = os.getenv('LLM_PROFESSION_API_URL', 'http://localhost:8000/get_profession_info/')
        self.roadmap_api_url = os.getenv('LLM_ROADMAP_API_URL', 'http://localhost:8000/get_profession_roadmap/')
        self.clean_api_url = os.getenv('LLM_CLEAN_API_URL', 'http://localhost:8000/clean_history/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    

    async def generate_response(self, message: str, user_id: int, parameters: dict = None) -> LLMResponse:
        """Отправляет сообщение в LLM API и возвращает ответ с профессиями"""
        if parameters is None:
            parameters = {}
        if not self.session:
            raise RuntimeError("LLMClient не инициализирован. Используйте async with.")
        
        try:
            payload = {
                "user_id": str(user_id),
                "prompt": message,
                "parameters": parameters
            }
            
           # Отправляем запрос к LLM API
            async with self.session.post(
                self.api_url,
                json=payload,
                #timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                                
                if response.status == 200:
                    result = await response.json()

                    msg = result.get("msg", "Извините, не удалось получить ответ.")
                    professions = result.get("professions", {})
                    test_info = result.get("test_info", {})

                    
                    # print(f"💬 Сообщение: {msg[:100]}...")
                    # print(f"💼 Профессии: {professions}")
                    
                    return LLMResponse(msg=msg, professions=professions, test_info=test_info)
                else:
                    error_text = await response.text()
                    print(f"❌ Ошибка API {response.status}: {error_text}")
                    return LLMResponse(msg="Извините, произошла ошибка при обработке запроса.")
                    
        except Exception as e:
            print(f"❌ Ошибка LLM API: {e}")
            import traceback
            traceback.print_exc()
            return LLMResponse(msg="Извините, произошла ошибка.")
    
    async def generate_start_message(self, user_id: int) -> LLMResponse:
        return await self.generate_response("/start", user_id)
    
    async def generate_help_message(self, user_id: int) -> LLMResponse:
        return await self.generate_response("/help", user_id)
    
    async def get_profession_info(self, profession_name: str, user_id: int) -> LLMResponse:
        """Получает подробную информацию о профессии через RAG систему"""
        if not self.session:
            raise RuntimeError("LLMClient не инициализирован. Используйте async with.")
        
        try:
            payload = {
                "user_id": str(user_id),
                "profession_name": profession_name,
            }
            
            async with self.session.post(
                self.profession_api_url,
                json=payload,
                #timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    msg = result.get("msg", "Извините, не удалось получить информацию о профессии.")
                    return LLMResponse(msg=msg, professions=None)
                else:
                    return LLMResponse(msg="Извините, произошла ошибка при получении информации о профессии.")
                    
        except Exception as e:
            print(f"Ошибка при получении информации о профессии: {e}")
            return LLMResponse(msg="Извините, произошла ошибка.")
        
    async def get_profession_roadmap(self, profession_name: str, user_id: int) -> LLMResponse:
        """Получает подробную информацию о курсах для профессии через RAG систему"""
        if not self.session:
            raise RuntimeError("LLMClient не инициализирован. Используйте async with.")
        
        try:
            payload = {
                "user_id": str(user_id),
                "profession_name": profession_name,
            }
            
            async with self.session.post(
                self.roadmap_api_url,
                json=payload,
                #timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    msg = result.get("msg", "Извините, не удалось получить информацию о профессии.")
                    return LLMResponse(msg=msg)
                else:
                    return LLMResponse(msg="Извините, произошла ошибка при получении информации о профессии.")
                    
        except Exception as e:
            print(f"Ошибка при получении roadmap профессии: {e}")
            return LLMResponse(msg="Извините, произошла ошибка.")

    async def clean_user_history(self, user_id: int):
        payload = {
            "user_id": str(user_id),
            "prompt": "prompt",
            "parameters": {}
        }

        # Отправляем запрос к LLM API
        async with self.session.post(
                self.clean_api_url,
                json=payload,
                # timeout=aiohttp.ClientTimeout(total=30)
        ) as response:

            if response.status == 200:
                return "Хоть сообщения и остались в чате, я забыл все, о чем мы общались."
            else:
                error_text = await response.text()
                print(f"❌ Ошибка API {response.status}: {error_text}")
                return "Извините, произошла ошибка при обработке запроса."