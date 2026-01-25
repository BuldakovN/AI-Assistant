import os
import json
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from professions_vector_index.yandex_embeddings import get_yandex_embeddings

# Пути для образовательной базы
DATA_PATH = os.path.join("data", "education", "education_comparison.json")
INDEX_DIR = os.path.join("data", "education", "education_vector")
DOCS_JSON = os.path.join(INDEX_DIR, "docs.json")


def load_educations(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def education_to_text(key: str, combined_str: str) -> Tuple[str, Dict]:
    title = key
    combined_text = (combined_str or "").strip()
    text = combined_text if combined_text else title
    if len(text) > 2000:
        text = text[:2300] + "..."
    metadata = {"key": key, "title": title}
    return text, metadata


def build_index() -> None:
    load_dotenv()
    os.makedirs(INDEX_DIR, exist_ok=True)
    data = load_educations(DATA_PATH)
    texts: List[str] = []
    metadatas: List[Dict] = []
    for key, obj in data.items():
        text, meta = education_to_text(key, obj)
        if text and len(text) > 10:
            texts.append(text)
            metadatas.append(meta)
    if not texts:
        raise RuntimeError("Не найдено валидных направлений для индексации")
    print(f"Найдено {len(texts)} направлений для индексации")
    print("Создание эмбеддингов с задержками для соблюдения лимитов API...")
    embeddings = get_yandex_embeddings()
    BATCH_SIZE = 5
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        print(f"Обрабатываем batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE}")
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        if i + BATCH_SIZE < len(texts):
            time.sleep(1.2)
    pairs = [
        (texts[i], list(map(float, all_embeddings[i]))) for i in range(len(texts))
    ]
    vs = FAISS.from_embeddings(text_embeddings=pairs, embedding=embeddings, metadatas=metadatas)
    vs.save_local(INDEX_DIR)
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump({i: m for i, m in enumerate(metadatas)}, f, ensure_ascii=False, indent=2)
    print(f"Индекс сохранён в: {INDEX_DIR}. Всего документов: {len(texts)}")

if __name__ == "__main__":
    build_index()
