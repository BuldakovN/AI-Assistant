"""Нормализация поля professions: LLM/tool_call иногда отдаёт JSON-строку вместо dict."""
from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def normalize_professions(prof: Any) -> Optional[Dict[str, Any]]:
    """
    Приводит professions к dict. Строку с JSON или Python-literal объектом пробует распарсить.
    Не удалось распознать — None.
    """
    if prof is None:
        return None
    if isinstance(prof, dict):
        return prof
    if isinstance(prof, str):
        s = prof.strip()
        if not s or s[0] not in "{[":
            return None
        try:
            parsed: Any = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                logger.warning(
                    "professions: не удалось разобрать строку как JSON или literal-dict (%s симв.)",
                    len(s),
                )
                return None
        return parsed if isinstance(parsed, dict) else None
    return None


def normalize_ai_recommendation_json(blob: Any) -> Any:
    """Если blob — dict и есть ключ professions, подменяет его на результат normalize_professions."""
    if not isinstance(blob, dict):
        return blob
    if "professions" not in blob:
        return blob
    out = dict(blob)
    out["professions"] = normalize_professions(out.get("professions"))
    return out
