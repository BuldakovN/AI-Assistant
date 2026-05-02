"""
Опциональная отправка трейсов в Arize Phoenix (OTLP).
Включается через PHOENIX_TRACING_ENABLED=1; register() должен выполниться до импорта LangChain.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, TypeVar

logger = logging.getLogger(__name__)

PHOENIX_TRACING_ACTIVE = False

R = TypeVar("R")


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


def _json_preview(obj: Any, max_chars: int = 16000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        s = repr(obj)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...(+{len(s) - max_chars} chars)"


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


def trace_yandex_chat_completion(
    *,
    model_name: str,
    messages: List[Dict[str, Any]],
    fn: Callable[[], R],
) -> R:
    if not PHOENIX_TRACING_ACTIVE:
        return fn()
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues, SpanAttributes

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("yandex.chat_completion") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
        span.set_attribute(SpanAttributes.INPUT_VALUE, _json_preview(messages))
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        try:
            text, tokens = fn()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _json_preview(text))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value)
        if tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(tokens))
        span.set_status(Status(StatusCode.OK))
        return text, tokens  # type: ignore[return-value]


def trace_yandex_tool_call(
    *,
    model_name: str,
    message: str,
    tools: List[Dict[str, Any]],
    fn: Callable[[], R],
) -> R:
    if not PHOENIX_TRACING_ACTIVE:
        return fn()
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues, SpanAttributes

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("yandex.tool_call") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            _json_preview({"message": message, "tools": tools}),
        )
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        try:
            result, tokens = fn()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _json_preview(result))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        if tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(tokens))
        span.set_status(Status(StatusCode.OK))
        return result, tokens  # type: ignore[return-value]
