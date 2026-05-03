"""
Абстракция для работы с LLM провайдерами через LangChain (включая Yandex).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from model.phoenix_tracing import trace_yandex_chat_completion, trace_yandex_tool_call

load_dotenv()


class LLMAdapter(ABC):
    """Абстрактный класс для работы с LLM"""

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    def chat_sync(self, messages: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Синхронный чат; возвращает (текст, completion_tokens)."""

    @abstractmethod
    def tool_call(
        self,
        message: str,
        tools: List[Dict],
        temperature: float = 0.6,
        max_tokens: int = 2000,
    ) -> Union[Optional[Dict[str, Any]], Tuple[Optional[Any], int]]:
        """Результат tool call или кортеж (result, completion_tokens)."""


class LangchainAdapter(LLMAdapter):
    """Все провайдеры, включая Yandex, через LangChain."""

    def __init__(self, provider: str = "openai", model_name: Optional[str] = None, **kwargs: Any):
        self.provider = provider.lower()
        self.model_name = model_name
        self.kwargs = kwargs
        self._chat_model: Any = None
        self._tracing_model_name: str = ""
        self._init_model()

    def _init_model(self) -> None:
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            api_key = self.kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = self.model_name or self.kwargs.get("model", "gpt-4o-mini")
            self._tracing_model_name = str(model)
            self._chat_model = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=self.kwargs.get("temperature", 0.5),
            )
        elif self.provider == "openrouter":
            from langchain_openrouter import ChatOpenRouter

            api_key = self.kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY должен быть установлен в переменных окружения")

            env_model = (os.getenv("OPENROUTER_MODEL") or "").strip()
            model = self.model_name or self.kwargs.get("model") or env_model or "openai/gpt-4o-mini"
            base_url = self.kwargs.get("base_url") or os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
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

            router_kwargs: Dict[str, Any] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": self.kwargs.get("temperature", 0.5),
            }
            if app_url:
                router_kwargs["app_url"] = app_url
            if app_title:
                router_kwargs["app_title"] = app_title

            self._tracing_model_name = str(model)
            self._chat_model = ChatOpenRouter(**router_kwargs)

        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            api_key = self.kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            model = self.model_name or self.kwargs.get("model", "claude-3-5-sonnet-20241022")
            self._tracing_model_name = str(model)
            self._chat_model = ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=self.kwargs.get("temperature", 0.5),
            )
        elif self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = self.kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
            model = self.model_name or self.kwargs.get("model", "gemini-pro")
            self._tracing_model_name = str(model)
            self._chat_model = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=self.kwargs.get("temperature", 0.5),
            )
        elif self.provider == "mistral":
            from langchain_mistralai import ChatMistralAI

            api_key = self.kwargs.get("api_key") or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY должен быть установлен в переменных окружения")
            model = self.model_name or self.kwargs.get("model", "mistral-small-latest")
            self._tracing_model_name = str(model)
            self._chat_model = ChatMistralAI(
                model=model,
                mistral_api_key=api_key,
                temperature=self.kwargs.get("temperature", 0.5),
            )
        elif self.provider == "yandex":
            from langchain_community.chat_models import ChatYandexGPT

            api_key = self.kwargs.get("api_key") or os.getenv("YANDEX_CLOUD_API_KEY")
            folder_id = self.kwargs.get("folder_id") or os.getenv("YANDEX_CLOUD_FOLDER")
            if not folder_id or not api_key:
                raise ValueError("YANDEX_CLOUD_FOLDER и YANDEX_CLOUD_API_KEY должны быть установлены")
            model = (
                self.model_name
                or self.kwargs.get("model")
                or os.getenv("YANDEX_CLOUD_MODEL", "yandexgpt-lite")
            )
            self._tracing_model_name = str(model)
            self._chat_model = ChatYandexGPT(
                api_key=api_key,
                folder_id=folder_id,
                model_name=model,
                temperature=self.kwargs.get("temperature", 0.5),
            )
        else:
            raise ValueError(
                f"Неподдерживаемый провайдер: {self.provider}. "
                f"Поддерживаются: openai, openrouter, anthropic, google, mistral, yandex"
            )

    @staticmethod
    def _coerce_message_text(raw: Any) -> str:
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

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        langchain_messages: List[Any] = []
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

    @staticmethod
    def _lc_response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts) if parts else str(content)
        return str(content)

    @staticmethod
    def _lc_completion_tokens(response: Any) -> int:
        um = getattr(response, "usage_metadata", None)
        if isinstance(um, dict):
            for key in ("output_tokens", "completion_tokens", "total_tokens"):
                v = um.get(key)
                if isinstance(v, int) and v >= 0:
                    return v
        rm = getattr(response, "response_metadata", None) or {}
        if isinstance(rm, dict):
            usage = rm.get("token_usage") or rm.get("usage") or {}
            if isinstance(usage, dict):
                for key in ("completion_tokens", "completionTokens", "output_tokens"):
                    v = usage.get(key)
                    if isinstance(v, int):
                        return v
        return 0

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        text, _tokens = self.chat_sync(messages)
        return text

    def chat_sync(self, messages: List[Dict[str, Any]]) -> Tuple[str, int]:
        langchain_messages = self._convert_messages(messages)

        def _run() -> Tuple[str, int]:
            response = self._chat_model.invoke(langchain_messages)
            return self._lc_response_text(response), self._lc_completion_tokens(response)

        if self.provider == "yandex":
            return trace_yandex_chat_completion(
                model_name=self._tracing_model_name,
                messages=messages,
                fn=_run,
            )
        return _run()

    def tool_call(
        self,
        message: str,
        tools: List[Dict],
        temperature: float = 0.6,
        max_tokens: int = 2000,
    ) -> Tuple[Optional[Any], int]:
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from pydantic import Field, create_model

        langchain_tools: List[Any] = []
        for yandex_tool in tools:
            if "function" not in yandex_tool:
                continue
            func_def = yandex_tool["function"]
            func_name = func_def.get("name", "")
            func_desc = func_def.get("description", "")
            params = func_def.get("parameters", {}).get("properties", {})
            fields: Dict[str, Any] = {}
            required = func_def.get("parameters", {}).get("required", [])

            for param_name, param_info in params.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                if param_type == "string":
                    field_type = str
                elif param_type == "integer":
                    field_type = int
                elif param_type == "number":
                    field_type = float
                elif param_type == "boolean":
                    field_type = bool
                else:
                    field_type = str

                if param_name in required:
                    fields[param_name] = (field_type, Field(description=param_desc))
                else:
                    fields[param_name] = (Optional[field_type], Field(default=None, description=param_desc))

            ParamsModel = create_model(f"{func_name.title()}Params", **fields)

            @tool(
                args_schema=ParamsModel,
                description=func_desc
                or "Structured tool: arguments are validated and returned unchanged.",
            )
            def dynamic_tool(**kwargs: Any) -> Any:
                return kwargs

            dynamic_tool.name = func_name or dynamic_tool.name

            langchain_tools.append(dynamic_tool)

        def _invoke_model(bound_model: Any) -> Tuple[Optional[Any], int]:
            response = bound_model.invoke([HumanMessage(content=message)])
            tokens = self._lc_completion_tokens(response)
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call0 = response.tool_calls[0]
                if isinstance(tool_call0, dict):
                    return tool_call0.get("args", {}), tokens
                if hasattr(tool_call0, "args"):
                    return tool_call0.args, tokens
                if hasattr(tool_call0, "get"):
                    return tool_call0.get("args", {}), tokens
            return None, tokens

        def _run() -> Tuple[Optional[Any], int]:
            try:
                bound = self._chat_model.bind(temperature=temperature, max_tokens=max_tokens)
            except TypeError:
                bound = self._chat_model
            model_with_tools = bound.bind_tools(langchain_tools)
            return _invoke_model(model_with_tools)

        if self.provider == "yandex":
            return trace_yandex_tool_call(
                model_name=self._tracing_model_name,
                message=message,
                tools=tools,
                fn=_run,
            )
        return _run()


def create_llm_adapter(provider: str = "yandex", **kwargs: Any) -> LLMAdapter:
    provider = provider.lower()
    if provider == "mistal":
        provider = "mistral"
    if provider not in ("yandex", "openai", "openrouter", "anthropic", "google", "mistral"):
        raise ValueError(
            f"Неподдерживаемый провайдер: {provider}. "
            f"Поддерживаются: yandex, openai, openrouter, anthropic, google, mistral"
        )
    return LangchainAdapter(
        provider=provider,
        model_name=kwargs.get("model_name"),
        **kwargs,
    )
