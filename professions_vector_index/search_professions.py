import os
import argparse
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from professions_vector_index.yandex_embeddings import get_yandex_embeddings


INDEX_DIR = os.path.join("data", "profession_vector")
COURSES_DIR = os.path.join("data/education", "education_vector")

INTERNAL_CONTEXT = (
    """
    Мария, 34 года
    Работает кассиром в банке. Не нравится, что нет индексации зарплаты уже несколько лет, не нравится коллектив на работе. Есть образование педагога психолога, но получала его для того, чтобы трудоустроиться на работу, где требовалось высшее образование. 
    Хотела бы сменить работу, но не знает, оставаться в этой же сфере или менять на другую. 
    У Марии много интересов, спорт, участвует в акциях по разделению вторсырья. Нравится чувствовать свое причастность к улучшению окружающей среды. 
    Хотела бы выйти на доход 60-80к. Чтобы у компании было ДМС, премирование сотрудников.
    """
).strip()


def search_top_k(query: str, k: int = 5) -> List[str]:
    load_dotenv()
    embeddings = get_yandex_embeddings()
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vs.similarity_search_with_score(query, k=k, threshold=0.75)
    results = []
    for d, score in docs:
        title = d.metadata.get("title") or d.metadata.get("key") or "?"
        results.append(title)
    return results

def rag_search(query: str, k: int = 5, api_key: str = None, folder_id: str = None, index_dir='INDEX_DIR') -> List[str]:
    directory = INDEX_DIR
    if index_dir == 'COURSES_DIR':
        directory = COURSES_DIR
    load_dotenv()
    embeddings = get_yandex_embeddings(api_key=api_key, folder_id=folder_id)
    vs = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
    docs = vs.similarity_search_with_score(query, k=k, threshold=0.75)
    return docs

def main():
    parser = argparse.ArgumentParser(description="Поиск топ-5 профессий по тексту пользователя")
    parser.add_argument("--text", type=str, required=False, default=None, help="Текстовый контекст пользователя. Если не задан, используется INTERNAL_CONTEXT в коде")
    parser.add_argument("--k", type=int, default=5, help="Сколько профессий вернуть")
    args = parser.parse_args()

    text = args.text or INTERNAL_CONTEXT
    if not text:
        print("Не задан текст (--text) и пустой INTERNAL_CONTEXT.")
        return

    titles = search_top_k(text, k=args.k)
    print("Топ профессий:")
    for i, t in enumerate(titles, 1):
        print(f"{i}. {t}")


if __name__ == "__main__":
    main()


