"""
Повторные вызовы LLM tool_call при сбоях парсинга/формата или исключениях адаптера.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _unwrap_raw(raw: Any) -> Tuple[Any, int]:
    if isinstance(raw, tuple):
        return raw[0], int(raw[1] if len(raw) > 1 else 0)
    return raw, 0


def _normalize_tool_result(result: Any) -> Tuple[Any, Optional[str]]:
    """
    Возвращает (result, reason_retry).
    reason_retry не None — нужен повтор (пустой ответ, битый JSON там, где ожидается JSON).
    """
    if result is None:
        return None, "empty_tool_result"

    if isinstance(result, str):
        s = result.strip()
        if not s:
            return None, "empty_tool_result"
        if s[0] in "[{":
            try:
                return json.loads(s), None
            except json.JSONDecodeError:
                return None, "invalid_json"
        return result, None

    return result, None


def invoke_tool_call_with_retries(
    adapter: Any,
    *,
    message: str,
    tools: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    log: Optional[logging.Logger] = None,
) -> Tuple[Any, int]:
    """
    Вызывает adapter.tool_call с ретраями.

    Повтор при: исключении из адаптера; пустом результате; строке, похожей на JSON, но с ошибкой парсинга.
    После исчерпания попыток — RuntimeError.
    """
    log = log or logger
    max_attempts = max(1, int(os.getenv("LLM_TOOL_CALL_MAX_ATTEMPTS", "4")))
    base_sleep = float(os.getenv("LLM_TOOL_CALL_RETRY_BACKOFF_SEC", "0.35"))

    last_exc: Optional[BaseException] = None
    last_soft: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = adapter.tool_call(
                message=message,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_exc = exc
            log.warning(
                "tool_call attempt %s/%s: exception from adapter: %s",
                attempt,
                max_attempts,
                exc,
            )
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"LLM tool_call failed after {max_attempts} attempts (adapter error)"
                ) from exc
            time.sleep(min(3.0, base_sleep * (2 ** (attempt - 1))))
            continue

        result, tokens = _unwrap_raw(raw)
        result, soft = _normalize_tool_result(result)

        if soft is None:
            return result, tokens

        last_soft = soft
        log.warning(
            "tool_call attempt %s/%s: %s, retrying",
            attempt,
            max_attempts,
            soft,
        )
        if attempt >= max_attempts:
            msg = f"LLM tool_call failed after {max_attempts} attempts: {soft}"
            raise RuntimeError(msg) from (last_exc if last_exc else None)

        time.sleep(min(3.0, base_sleep * (2 ** (attempt - 1))))
