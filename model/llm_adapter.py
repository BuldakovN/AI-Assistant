"""
Абстракция для работы с различными LLM провайдерами через единый интерфейс.
Поддерживает Yandex Cloud и Langchain провайдеры.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv

from model.phoenix_tracing import trace_yandex_chat_completion, trace_yandex_tool_call

load_dotenv()


class LLMAdapter(ABC):
    """Абстрактный класс для работы с LLM"""
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Выполняет чат-запрос к LLM
        
        Args:
            messages: Список сообщений в формате [{"role": "system|user|assistant", "text": "..."}]
            
        Returns:
            Текст ответа от LLM
        """
        pass
    
    @abstractmethod
    def chat_sync(self, messages: List[Dict[str, str]]) -> str:
        """
        Синхронная версия chat для случаев, когда async не нужен
        
        Args:
            messages: Список сообщений в формате [{"role": "system|user|assistant", "text": "..."}]
            
        Returns:
            Текст ответа от LLM
        """
        pass
    
    @abstractmethod
    def tool_call(self, message: str, tools: List[Dict], temperature: float = 0.6, max_tokens: int = 2000) -> Optional[Dict]:
        """
        Выполняет запрос с использованием tools (function calling)
        
        Args:
            message: Текст сообщения пользователя
            tools: Список инструментов в формате Yandex Cloud tools
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            
        Returns:
            Словарь с аргументами вызванной функции или None
        """
        pass


class YandexAdapter(LLMAdapter):
    """Адаптер для работы с Yandex Cloud ML SDK"""
    
    def __init__(
        self,
        folder_id: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        from yandex_cloud_ml_sdk import YCloudML
        
        self.folder_id = folder_id or os.getenv('YANDEX_CLOUD_FOLDER', '')
        self.api_key = api_key or os.getenv('YANDEX_CLOUD_API_KEY', '')
        self.model_name = model_name or os.getenv('YANDEX_CLOUD_MODEL', 'yandexgpt-5-lite')
        
        if not self.folder_id or not self.api_key:
            raise ValueError("YANDEX_CLOUD_FOLDER и YANDEX_CLOUD_API_KEY должны быть установлены в переменных окружения")
        
        sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        model_uri = f"gpt://{self.folder_id}/{self.model_name}"
        self.model = sdk.models.completions(model_uri)
        self.model = self.model.configure(temperature=0.5)
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Асинхронная версия chat (для Yandex SDK синхронная)"""
        return self.chat_sync(messages)
    
    def chat_sync(self, messages: List[Dict[str, str]]) -> (str, int):
        """Синхронный чат через Yandex SDK"""
        # Yandex SDK ожидает список словарей с "role" и "text"

        def _run():
            response = self.model.run(messages)
            return response.alternatives[0].text, response.usage.completion_tokens

        return trace_yandex_chat_completion(model_name=self.model_name, messages=messages, fn=_run)
    
    def tool_call(self, message: str, tools: List[Dict], temperature: float = 0.6, max_tokens: int = 2000) -> Optional[Tuple]:
        """Выполняет tool call через Yandex API"""
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-folder-id": self.folder_id
        }
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_name}",
            "completionOptions": {
                "temperature": temperature,
                "maxTokens": max_tokens
            },
            "tools": tools,
            "messages": [
                {
                    "role": "user",
                    "text": message
                }
            ]
        }

        def _run():
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()['result']['alternatives'][0]['message']
            llm_tokens = response.json()['result']['usage']['completionTokens']
            if result.get('toolCallList'):
                return result['toolCallList']['toolCalls'][0]['functionCall']['arguments'], int(llm_tokens)
            return None, 0

        return trace_yandex_tool_call(
            model_name=self.model_name,
            message=message,
            tools=tools,
            fn=_run,
        )


class LangchainAdapter(LLMAdapter):
    """Адаптер для работы с Langchain провайдерами"""
    
    def __init__(self, provider: str = "openai", model_name: Optional[str] = None, **kwargs):
        """
        Инициализирует Langchain адаптер
        
        Args:
            provider: Название провайдера ("openai", "openrouter", "anthropic", "google", "mistral", "yandex", etc.)
            model_name: Название модели (если None, для openrouter — ``OPENROUTER_MODEL`` или ``OPENROUTER_BASE_URL``)
            **kwargs: Дополнительные параметры для инициализации (api_key, temperature, etc.)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.kwargs = kwargs
        self._llm = None
        self._chat_model = None
        self._init_model()
        print('Provider:', provider)
        print('Model name:', model_name)
    
    def _init_model(self):
        """Инициализирует модель в зависимости от провайдера"""
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = self.kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            model = self.model_name or self.kwargs.get('model', 'gpt-4o-mini')
            self._chat_model = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=self.kwargs.get('temperature', 0.5)
            )
        elif self.provider == "openrouter":
            from langchain_openrouter import ChatOpenRouter
            api_key = self.kwargs.get('api_key') or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY должен быть установлен в переменных окружения")

            env_model = (os.getenv("OPENROUTER_MODEL") or "").strip()
            # Дефолт как в env.example; бесплатные модели часто дают 400 Provider returned error
            model = self.model_name or self.kwargs.get("model") or env_model or "openai/gpt-4o-mini"
            base_url = self.kwargs.get('base_url') or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
            # ChatOpenRouter ожидает app_url / app_title; default_headers уходит в model_kwargs
            # и ломает вызов OpenRouter SDK (Chat.send не принимает default_headers).
            app_url = (
                self.kwargs.get("app_url")
                or os.getenv("OPENROUTER_APP_URL")
                or self.kwargs.get("http_referer")
                or os.getenv("OPENROUTER_HTTP_REFERER")
            )
            app_title = (
                self.kwargs.get("app_title")
                or os.getenv("OPENROUTER_APP_TITLE")
                or self.kwargs.get("x_title")
                or os.getenv("OPENROUTER_X_TITLE")
            )

            print("Base URL:", base_url)
            print("OpenRouter model id:", model)

            router_kwargs = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": self.kwargs.get("temperature", 0.5),
            }
            if app_url:
                router_kwargs["app_url"] = app_url
            if app_title:
                router_kwargs["app_title"] = app_title

            self._chat_model = ChatOpenRouter(**router_kwargs)
        
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = self.kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            model = self.model_name or self.kwargs.get('model', 'claude-3-5-sonnet-20241022')
            self._chat_model = ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=self.kwargs.get('temperature', 0.5)
            )
        elif self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = self.kwargs.get('api_key') or os.getenv('GOOGLE_API_KEY')
            model = self.model_name or self.kwargs.get('model', 'gemini-pro')
            self._chat_model = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=self.kwargs.get('temperature', 0.5)
            )
        elif self.provider == "mistral":
            from langchain_mistralai import ChatMistralAI
            api_key = self.kwargs.get('api_key') or os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("MISTRAL_API_KEY должен быть установлен в переменных окружения")
            model = self.model_name or self.kwargs.get('model', 'mistral-small-latest')
            self._chat_model = ChatMistralAI(
                model=model,
                mistral_api_key=api_key,
                temperature=self.kwargs.get('temperature', 0.5)
            )
        elif self.provider == "yandex":
            from langchain_community.chat_models import ChatYandexGPT
            api_key = self.kwargs.get('api_key') or os.getenv('YANDEX_CLOUD_API_KEY')
            folder_id = self.kwargs.get('folder_id') or os.getenv('YANDEX_CLOUD_FOLDER')
            self._chat_model = ChatYandexGPT(
                api_key=api_key,
                folder_id=folder_id,
                temperature=self.kwargs.get('temperature', 0.5)
            )
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {self.provider}. "
                           f"Поддерживаются: openai, openrouter, anthropic, google, mistral, yandex")

    @staticmethod
    def _coerce_message_text(raw: Any) -> str:
        """Текст сообщения для LangChain: str или OpenAI-style list блоков из CRUD/истории."""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts: List[str] = []
            for block in raw:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block.get("content"), str):
                        parts.append(block["content"])
            return "\n".join(parts)
        return str(raw)
    
    def _convert_messages(self, messages: List[Dict[str, Any]]):
        """Конвертирует сообщения из формата приложения в формат Langchain"""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            role = (msg.get("role") or "user").strip().lower()
            text = self._coerce_message_text(msg.get("text"))
            if not text.strip():
                text = self._coerce_message_text(msg.get("content"))
            text = text.strip()
            if not text:
                continue

            if role == "system":
                langchain_messages.append(SystemMessage(content=text))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=text))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=text))
            else:
                langchain_messages.append(HumanMessage(content=text))

        if not langchain_messages:
            langchain_messages.append(HumanMessage(content="."))

        return langchain_messages
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Асинхронный чат через Langchain"""
        langchain_messages = self._convert_messages(messages)
        response = await self._chat_model.ainvoke(langchain_messages)
        return response.content
    
    def chat_sync(self, messages: List[Dict[str, str]]) -> (str, int):
        """Синхронный чат через Langchain"""
        langchain_messages = self._convert_messages(messages)
        response = self._chat_model.invoke(langchain_messages)
        return response.content, 0
    
    def tool_call(self, message: str, tools: List[Dict], temperature: float = 0.6, max_tokens: int = 2000) -> Optional[Dict]:
        """
        Выполняет tool call через Langchain.
        Конвертирует Yandex tools формат в Langchain tools формат.
        """
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from pydantic import Field, create_model
        
        # Конвертируем Yandex tools в Langchain tools
        langchain_tools = []
        for yandex_tool in tools:
            if 'function' in yandex_tool:
                func_def = yandex_tool['function']
                func_name = func_def.get('name', '')
                func_desc = func_def.get('description', '')
                params = func_def.get('parameters', {}).get('properties', {})
                
                # Создаем Pydantic модель для параметров
                fields = {}
                required = func_def.get('parameters', {}).get('required', [])
                
                for param_name, param_info in params.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    
                    # Маппинг типов
                    if param_type == 'string':
                        field_type = str
                    elif param_type == 'integer':
                        field_type = int
                    elif param_type == 'number':
                        field_type = float
                    elif param_type == 'boolean':
                        field_type = bool
                    else:
                        field_type = str
                    
                    if param_name in required:
                        fields[param_name] = (field_type, Field(description=param_desc))
                    else:
                        fields[param_name] = (Optional[field_type], Field(default=None, description=param_desc))
                
                # Создаем динамический класс для параметров (Pydantic v2)
                ParamsModel = create_model(f"{func_name.title()}Params", **fields)
                
                # Создаем tool
                @tool(args_schema=ParamsModel)
                def dynamic_tool(**kwargs):
                    """Dynamic tool wrapper."""
                    return kwargs

                # Для совместимости с версиями langchain_core,
                # где tool() не принимает name/description.
                dynamic_tool.name = func_name or dynamic_tool.name
                if func_desc:
                    dynamic_tool.description = func_desc

                langchain_tools.append(dynamic_tool)
        
        # Биндим tools к модели
        model_with_tools = self._chat_model.bind_tools(langchain_tools)
        
        # Выполняем запрос
        response = model_with_tools.invoke([HumanMessage(content=message)])
        
        # Извлекаем результат tool call
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            # tool_calls может быть списком объектов ToolCall или словарей
            if isinstance(tool_call, dict):
                return tool_call.get('args', {})
            elif hasattr(tool_call, 'args'):
                return tool_call.args
            elif hasattr(tool_call, 'get'):
                return tool_call.get('args', {})
        
        return None


def create_llm_adapter(provider: str = "yandex", **kwargs) -> LLMAdapter:
    """
    Фабрика для создания LLM адаптера
    
    Args:
        provider: Провайдер LLM ("yandex", "openai", "openrouter", "anthropic", "google", "mistral")
        **kwargs: Дополнительные параметры для инициализации адаптера
        
    Returns:
        Экземпляр LLMAdapter
    """
    provider = provider.lower()
    
    if provider == "yandex":
        return YandexAdapter(
            folder_id=kwargs.get('folder_id'),
            api_key=kwargs.get('api_key'),
            model_name=kwargs.get('model_name'),
        )
    elif provider in ["openai", "openrouter", "anthropic", "google", "mistral"]:
        return LangchainAdapter(
            provider=provider,
            model_name=kwargs.get('model_name'),
            **kwargs
        )
    else:
        raise ValueError(f"Неподдерживаемый провайдер: {provider}. "
                       f"Поддерживаются: yandex, openai, openrouter, anthropic, google, mistral")

