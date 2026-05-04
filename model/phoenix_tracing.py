"""
Опциональная отправка трейсов в Arize Phoenix (OTLP).
Включается через PHOENIX_TRACING_ENABLED=1; register() должен выполниться до импорта LangChain.

Вызовы LLM идут через LangChain (все провайдеры, включая Yandex); те же цепочки
покрываются ``openinference-instrumentation-langchain`` при ``auto_instrument=True``.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PHOENIX_TRACING_ACTIVE = False


def _env_flag(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _collector_endpoint_for_protocol(endpoint: str, protocol: str) -> str:
    """
    Для OTLP HTTP Phoenix ожидает POST на .../v1/traces (см. phoenix.otel — без пути экспортер бьёт в / → 405).
    gRPC: authority :4317 без /v1/traces.
    """
    base = endpoint.strip().rstrip("/")
    if protocol != "http/protobuf":
        return base
    if base.endswith("/v1/traces"):
        return base
    return f"{base}/v1/traces"


def setup_phoenix_if_enabled() -> None:
    """Регистрирует OTLP exporter и auto-instrument (LangChain и др.), если задан флаг."""
    global PHOENIX_TRACING_ACTIVE
    if PHOENIX_TRACING_ACTIVE:
        return
    if not _env_flag("PHOENIX_TRACING_ENABLED"):
        return
    try:
        from phoenix.otel import register
    except ImportError:
        logger.warning(
            "PHOENIX_TRACING_ENABLED=1, но пакет arize-phoenix-otel не установлен — трейсы отключены"
        )
        return

    raw_endpoint = (os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006").strip()
    project_name = (os.getenv("PHOENIX_PROJECT_NAME") or "ai-gigaschool-llm").strip()
    raw_proto = (os.getenv("PHOENIX_OTLP_PROTOCOL") or "http/protobuf").strip().lower()
    protocol: str = raw_proto if raw_proto in ("http/protobuf", "grpc") else "http/protobuf"

    endpoint = _collector_endpoint_for_protocol(raw_endpoint, protocol)

    try:
        register(
            endpoint=endpoint,
            project_name=project_name,
            auto_instrument=True,
            batch=True,
            verbose=False,
            protocol=protocol,  # type: ignore[arg-type]
        )
    except Exception:
        logger.exception("Не удалось инициализировать Phoenix OTLP (endpoint=%s)", endpoint)
        return

    PHOENIX_TRACING_ACTIVE = True
    logger.info(
        "Phoenix tracing: endpoint=%s project=%s protocol=%s",
        endpoint,
        project_name,
        protocol,
    )
