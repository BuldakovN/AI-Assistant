"""
Только вызовы LLM: чат и tool_call. Конфигурация инструментов — из model/config.yaml.
"""
import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from common.error_logging import setup_service_error_logging
from model.llm_adapter import create_llm_adapter
from model.tool_call_retry import invoke_tool_call_with_retries

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
setup_service_error_logging("llm-service")

# Отдельный файл только для audit вызовов LLM (не смешивается с errors.log / stdout).
_calls_audit_logger = logging.getLogger("llm_service.calls_audit")
_calls_audit_handler_installed = False
_calls_audit_handler_lock = threading.Lock()

LLMCallLogLevel = Literal["off", "summary", "full"]


def _llm_call_log_level() -> LLMCallLogLevel:
    raw = (os.getenv("LLM_CALL_LOG") or "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return "off"
    if raw in ("1", "true", "yes", "on", "summary"):
        return "summary"
    if raw == "full":
        return "full"
    return "off"


def _llm_call_log_text_max() -> int:
    try:
        return max(100, int(os.getenv("LLM_CALL_LOG_TEXT_MAX", "2000")))
    except ValueError:
        return 2000


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"... (+{len(s) - limit} chars)"


def _resolved_log_root() -> Path:
    raw = (os.environ.get("LOG_ROOT") or "").strip()
    return Path(raw) if raw else Path.cwd() / "logs"


def _ensure_llm_calls_audit_file_handler() -> None:
    """Один раз подключает RotatingFileHandler к audit-логгеру, если включён LLM_CALL_LOG."""
    global _calls_audit_handler_installed
    if _calls_audit_handler_installed:
        return
    if _llm_call_log_level() == "off":
        return
    with _calls_audit_handler_lock:
        if _calls_audit_handler_installed:
            return
        for h in _calls_audit_logger.handlers:
            if getattr(h, "_llm_calls_audit_file", None):
                _calls_audit_handler_installed = True
                return
        explicit = (os.getenv("LLM_CALL_LOG_FILE") or "").strip()
        if explicit:
            log_path = Path(explicit)
        else:
            log_path = _resolved_log_root() / "llm-service" / "llm_calls.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=8,
            encoding="utf-8",
        )
        handler._llm_calls_audit_file = True  # type: ignore[attr-defined]
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        _calls_audit_logger.addHandler(handler)
        _calls_audit_logger.setLevel(logging.INFO)
        _calls_audit_logger.propagate = False
        _calls_audit_handler_installed = True


def _audit_log_line(message: str, *args: Any) -> None:
    _ensure_llm_calls_audit_file_handler()
    if _llm_call_log_level() == "off":
        return
    _calls_audit_logger.info(message, *args)


def _log_llm_chat(level: LLMCallLogLevel, messages: List[Dict[str, Any]], tokens: int, ms: float) -> None:
    if level == "off":
        return
    n = len(messages)
    total_chars = 0
    for m in messages:
        t = m.get("text") if isinstance(m.get("text"), str) else m.get("content")
        if isinstance(t, str):
            total_chars += len(t)
        elif t is not None:
            total_chars += len(str(t))
    extra: Dict[str, Any] = {
        "op": "chat",
        "message_count": n,
        "approx_input_chars": total_chars,
        "completion_tokens": tokens,
        "duration_ms": round(ms, 2),
    }
    if level == "full":
        lim = _llm_call_log_text_max()
        extra["messages_preview"] = [
            {
                "role": m.get("role"),
                "text": _truncate(
                    m.get("text") if isinstance(m.get("text"), str) else str(m.get("content", "")),
                    lim,
                ),
            }
            for m in messages
        ]
    _audit_log_line("LLM call %s", json.dumps(extra, ensure_ascii=False))


def _log_llm_tool_call(
    level: LLMCallLogLevel,
    tool_key: str,
    message: str,
    tokens: int,
    ms: float,
    *,
    result_ok: bool,
) -> None:
    if level == "off":
        return
    extra: Dict[str, Any] = {
        "op": "tool_call",
        "tool_key": tool_key,
        "message_chars": len(message or ""),
        "completion_tokens": tokens,
        "duration_ms": round(ms, 2),
        "result_ok": result_ok,
    }
    if level == "full":
        extra["message_preview"] = _truncate(message or "", _llm_call_log_text_max())
    _audit_log_line("LLM call %s", json.dumps(extra, ensure_ascii=False))


folder_id = os.getenv("YANDEX_CLOUD_FOLDER", "")
api_key = os.getenv("YANDEX_CLOUD_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "yandex").lower()

config_path = Path(__file__).parent / "config.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


config = _load_yaml(config_path)


def _extra_exception_log_text(exc: BaseException) -> str:
    """Тело ответа OpenRouter SDK / httpx, если есть (иначе generic Provider returned error)."""
    chunks: List[str] = []
    body = getattr(exc, "body", None)
    if isinstance(body, str) and body.strip():
        chunks.append(body.strip()[:8000])
    raw = getattr(exc, "raw_response", None)
    if raw is not None:
        try:
            t = raw.text
            if isinstance(t, str) and t.strip():
                chunks.append(f"raw_response={t.strip()[:8000]}")
        except Exception:
            pass
    data = getattr(exc, "data", None)
    if data is not None:
        try:
            chunks.append(f"data={data!r}")
        except Exception:
            pass
    return " | ".join(chunks)


if LLM_PROVIDER == "yandex":
    _adapter = create_llm_adapter(
        provider="yandex",
        folder_id=folder_id,
        api_key=api_key,
    )
else:
    _adapter = create_llm_adapter(provider=LLM_PROVIDER)

app = FastAPI(title="LLM Service", version="0.1.0")
Instrumentator().instrument(app).expose(app)


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    text: str
    tokens: int = 0


class ToolCallRequest(BaseModel):
    message: str
    tool_key: str = Field(..., description="Ключ набора tools в config.yaml, например tools, make_json_tool")
    temperature: float = 0.6
    max_tokens: int = 2000


class ToolCallResponse(BaseModel):
    result: Optional[Any] = None
    tokens: int = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    log_level = _llm_call_log_level()
    t0 = time.perf_counter()
    try:
        text, tokens = _adapter.chat_sync(req.messages)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _log_llm_chat(log_level, req.messages, int(tokens or 0), dt_ms)
        if log_level == "full":
            lim = _llm_call_log_text_max()
            _audit_log_line(
                "LLM call chat_reply %s",
                json.dumps(
                    {"reply_preview": _truncate(text or "", lim), "completion_tokens": int(tokens or 0)},
                    ensure_ascii=False,
                ),
            )
        return ChatResponse(text=text, tokens=int(tokens or 0))
    except Exception as e:
        extra = _extra_exception_log_text(e)
        if extra:
            logger.error("Детали ответа провайдера: %s", extra)
        logger.exception("Ошибка /v1/chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/tool_call", response_model=ToolCallResponse)
def tool_call(req: ToolCallRequest):
    if req.tool_key not in config:
        raise HTTPException(status_code=400, detail=f"Unknown tool_key: {req.tool_key}")
    tools = config[req.tool_key]
    log_level = _llm_call_log_level()
    t0 = time.perf_counter()
    try:
        result, tokens = invoke_tool_call_with_retries(
            _adapter,
            message=req.message,
            tools=tools,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            log=logger,
        )
    except RuntimeError as e:
        logger.exception("Ошибка /v1/tool_call после ретраев: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    dt_ms = (time.perf_counter() - t0) * 1000.0
    _log_llm_tool_call(
        log_level,
        req.tool_key,
        req.message,
        int(tokens or 0),
        dt_ms,
        result_ok=True,
    )
    if log_level == "full" and result is not None:
        lim = _llm_call_log_text_max()
        try:
            preview = json.dumps(result, ensure_ascii=False)
        except TypeError:
            preview = repr(result)
        _audit_log_line(
            "LLM call tool_result %s",
            json.dumps({"tool_key": req.tool_key, "result_preview": _truncate(preview, lim)}, ensure_ascii=False),
        )

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass

    return ToolCallResponse(result=result, tokens=int(tokens or 0))
