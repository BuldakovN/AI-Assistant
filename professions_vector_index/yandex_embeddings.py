import os
from typing import Optional

from langchain_community.embeddings.yandex import YandexGPTEmbeddings


def get_yandex_embeddings(api_key: Optional[str] = None, folder_id: Optional[str] = None) -> YandexGPTEmbeddings:
    key = api_key or os.getenv("YANDEX_CLOUD_API_KEY")
    folder = folder_id or os.getenv("YANDEX_CLOUD_FOLDER")
    if not key or not folder:
        raise ValueError("Не заданы YANDEX_CLOUD_API_KEY и/или YANDEX_CLOUD_FOLDER. Укажите их в окружении.")
    return YandexGPTEmbeddings(api_key=key, folder_id=folder)


