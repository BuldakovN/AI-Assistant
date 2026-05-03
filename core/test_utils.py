"""Утилиты для сопоставления имён тестов с каталогом prof_tests."""
from __future__ import annotations

from typing import Any, Dict, Optional


def canonical_prof_test_key(prof_tests: Dict[str, Any], raw: Optional[str]) -> str:
    """Имя теста от LLM → ключ в prof_tests.yaml (синонимы из старых промптов/enum)."""
    if not isinstance(raw, str):
        return ""
    name = raw.strip()
    keys = (prof_tests.get("test_questions") or {}).keys()
    if name in keys:
        return name
    aliases = {
        "Якоря карьеры Шейна": "Тест Якоря карьеры Шейна",
        "Тест Холланда (RIASEC)": "Тест Холланда/RIASEC",
        "MBTI": "Тест MBTI",
        "PCM": "Тест PCM",
    }
    mapped = aliases.get(name)
    if mapped and mapped in keys:
        return mapped
    return name


def convert_to_qa_format(data_list: list) -> str:
    result_lines: list[str] = []
    for item in data_list:
        question = item[0]
        answer = item[1]
        result_lines.append(f"Вопрос: {question}")
        result_lines.append(f"Ответ пользователя: {answer}")
        result_lines.append("")
    return "\n".join(result_lines)
