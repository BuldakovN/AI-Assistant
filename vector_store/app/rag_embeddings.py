"""
Эмбеддинги для RAG (FAISS по профессиям и курсам).

Выбор провайдера — как у ``create_llm_adapter`` в ``model/llm_adapter.py``:
``RAG_EMBEDDING_PROVIDER`` (yandex | openai | google | mistral | openrouter). Если не задан —
берётся ``LLM_PROVIDER``. Провайдер **anthropic** для LLM поддерживается, для эмбеддингов
публичного API нет — задайте другой ``RAG_EMBEDDING_PROVIDER`` и пересоберите индекс.

Переменные окружения (дополнительно к ключам провайдера):

**Yandex** — ``YANDEX_CLOUD_API_KEY``, ``YANDEX_CLOUD_FOLDER``; опционально
``RAG_YANDEX_EMBEDDING_QUERY_MODEL``, ``RAG_YANDEX_EMBEDDING_DOC_MODEL``,
``RAG_YANDEX_EMBEDDING_MODEL_VERSION``.

**OpenAI** — ``OPENAI_API_KEY``; опционально ``RAG_OPENAI_EMBEDDING_MODEL`` (по умолчанию
``text-embedding-3-small``).

**Google** — ``GOOGLE_API_KEY``; опционально ``RAG_GOOGLE_EMBEDDING_MODEL``.

**Mistral** — ``MISTRAL_API_KEY``; опционально ``RAG_MISTRAL_EMBEDDING_MODEL`` (по умолчанию
``mistral-embed``).

**OpenRouter** — ``OPENROUTER_API_KEY``; опционально ``OPENROUTER_BASE_URL`` (по умолчанию
``https://openrouter.ai/api/v1``), ``OPENROUTER_HTTP_REFERER``, ``OPENROUTER_X_TITLE``;
модель эмбеддингов: ``RAG_OPENROUTER_EMBEDDING_MODEL`` (по умолчанию
``openai/text-embedding-3-small`` — см. каталог embedding-моделей на openrouter.ai).
Для OpenRouter в ``OpenAIEmbeddings`` отключена проверка длины через tiktoken
(``check_embedding_ctx_length=False``): иначе LangChain шлёт в API массивы token id,
OpenRouter на этом даёт пустой ``data`` и ошибка «No embedding data received».
Тексты в скриптах сборки индекса уже укорочены по символам.

Важно: вектора в FAISS должны быть посчитаны тем же провайдером и совместимой моделью,
что и при поиске; смена провайдера требует пересборки индексов.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.embeddings import Embeddings


def _llm_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "yandex").strip().lower()


def _rag_embedding_provider() -> str:
    explicit = (os.getenv("RAG_EMBEDDING_PROVIDER") or "").strip().lower()
    return explicit if explicit else _llm_provider()


def _build_yandex_embeddings(
    api_key: Optional[str] = None,
    folder_id: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
    from langchain_community.embeddings.yandex import YandexGPTEmbeddings

    key = api_key or kwargs.get("api_key") or os.getenv("YANDEX_CLOUD_API_KEY")
    folder = folder_id or kwargs.get("folder_id") or os.getenv("YANDEX_CLOUD_FOLDER")
    if not key or not folder:
        raise ValueError(
            "Не заданы YANDEX_CLOUD_API_KEY и/или YANDEX_CLOUD_FOLDER для эмбеддингов Yandex."
        )

    query_model = (os.getenv("RAG_YANDEX_EMBEDDING_QUERY_MODEL") or "").strip()
    doc_model = (os.getenv("RAG_YANDEX_EMBEDDING_DOC_MODEL") or "").strip()
    version_raw = (os.getenv("RAG_YANDEX_EMBEDDING_MODEL_VERSION") or "").strip()

    ykwargs: dict[str, Any] = {"api_key": key, "folder_id": folder}
    if query_model:
        ykwargs["model_name"] = query_model
    if doc_model:
        ykwargs["doc_model_name"] = doc_model
    if version_raw:
        ykwargs["model_version"] = version_raw

    return YandexGPTEmbeddings(**ykwargs)


def _build_openai_embeddings(**kwargs: Any) -> Embeddings:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "Для OpenAI embeddings установите зависимости: pip install langchain-openai openai"
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
    model = (
        kwargs.get("model_name")
        or os.getenv("RAG_OPENAI_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    model = str(model).strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан для эмбеддингов OpenAI (RAG).")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def _build_google_embeddings(**kwargs: Any) -> Embeddings:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "Для Google embeddings установите: pip install langchain-google-genai"
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
    model = (
        kwargs.get("model_name")
        or os.getenv("RAG_GOOGLE_EMBEDDING_MODEL")
        or "models/text-embedding-004"
    )
    model = str(model).strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY не задан для эмбеддингов Google (RAG).")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def _build_mistral_embeddings(**kwargs: Any) -> Embeddings:
    try:
        from langchain_mistralai import MistralAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "Для Mistral embeddings установите: pip install langchain-mistralai"
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("MISTRAL_API_KEY")
    model = (
        kwargs.get("model_name")
        or os.getenv("RAG_MISTRAL_EMBEDDING_MODEL")
        or "mistral-embed"
    )
    model = str(model).strip()
    if not api_key:
        raise ValueError("MISTRAL_API_KEY не задан для эмбеддингов Mistral (RAG).")
    return MistralAIEmbeddings(api_key=api_key, model=model)


def _build_openrouter_embeddings(**kwargs: Any) -> Embeddings:
    """OpenAI-совместимый /v1/embeddings через OpenRouter (как ChatOpenAI в llm_adapter)."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "Для OpenRouter embeddings установите зависимости: pip install langchain-openai openai"
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY не задан для эмбеддингов OpenRouter (RAG).")

    base_url = (
        kwargs.get("base_url")
        or os.getenv("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    base_url = str(base_url).strip().rstrip("/")

    model = (
        kwargs.get("model_name")
        or os.getenv("RAG_OPENROUTER_EMBEDDING_MODEL")
        or "openai/text-embedding-3-small"
    )
    model = str(model).strip()

    http_referer = kwargs.get("http_referer") or os.getenv("OPENROUTER_HTTP_REFERER")
    x_title = kwargs.get("x_title") or os.getenv("OPENROUTER_X_TITLE")

    default_headers: dict[str, str] = {}
    if http_referer:
        default_headers["HTTP-Referer"] = http_referer
    if x_title:
        default_headers["X-Title"] = x_title

    emb_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        # OpenRouter ожидает строки/array of strings в `input`, а не token ids из
        # _get_len_safe_embeddings (иначе HTTP 200 и пустой `data` → SDK: No embedding data).
        "check_embedding_ctx_length": False,
    }
    if default_headers:
        emb_kwargs["default_headers"] = default_headers

    return OpenAIEmbeddings(**emb_kwargs)


def create_rag_embeddings(provider: Optional[str] = None, **kwargs: Any) -> Embeddings:
    """
    Фабрика эмбеддингов для RAG по тому же набору провайдеров, что и ``create_llm_adapter``
    (кроме anthropic — для него нет embeddings API).

    Args:
        provider: yandex | openai | google | mistral | openrouter. По умолчанию
            RAG_EMBEDDING_PROVIDER или LLM_PROVIDER.
        **kwargs: api_key, folder_id (Yandex), model_name и т.д.
    """
    p = (provider or _rag_embedding_provider()).strip().lower()

    if p == "anthropic":
        raise ValueError(
            "Провайдер «anthropic» не предоставляет публичный API эмбеддингов. "
            "Задайте RAG_EMBEDDING_PROVIDER в {yandex, openai, google, mistral, openrouter} "
            "и пересоберите FAISS под выбранную модель."
        )

    if p == "yandex":
        return _build_yandex_embeddings(**kwargs)

    if p == "openai":
        return _build_openai_embeddings(**kwargs)

    if p == "google":
        return _build_google_embeddings(**kwargs)

    if p == "mistral":
        return _build_mistral_embeddings(**kwargs)

    if p == "openrouter":
        return _build_openrouter_embeddings(**kwargs)

    raise ValueError(
        f"Неподдерживаемый провайдер эмбеддингов: {p!r}. "
        "Как в model/llm_adapter.create_llm_adapter: yandex, openai, openrouter, google, mistral; "
        "anthropic — только для LLM, для RAG выберите другой RAG_EMBEDDING_PROVIDER."
    )


def get_rag_embeddings(
    api_key: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> Embeddings:
    """Совместимость с вызовами из поиска / сборки индекса (ключи Yandex)."""
    return create_rag_embeddings(api_key=api_key, folder_id=folder_id)
