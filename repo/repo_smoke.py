import os
from repository import Repository, RepositoryConfig

def main():
    # чтобы не портить боевую БД — отдельный файл
    db_url = os.getenv("SQLITE_URL", "sqlite:///test_app.sqlite3")
    repo = Repository(RepositoryConfig(db_url=db_url, echo=False))
    repo.create_schema()

    user_id = "574851787"

    about_payload = {
        user_id: {
            "user_metadata": {
                "user_state": "recommendation",
                "user_type": "school",
                "who_user": "Пользователь — школьница Ира, 16 лет.",
                "about_user": "Любит английский, русский, литературу.",
                "recommended_test": "RIASEC + Якоря Шейна",
                "test_user": "Коммуникабельная...",
                "ai_recommendation_json": {
                    "professions": {"Переводчик": "Переводит тексты"}
                },
                "ai_recommendation": "РЕКОМЕНДОВАННЫЕ ПРОФЕССИИ: ..."
            }
        }
    }

    about_data_payload = [
        {"role": "system", "text": "текст игр"},
        {"role": "user", "text": "тонус"},
        {"role": "assistant", "text": "ок"}
    ]

    test_data_payload = [
        {"role": "system", "text": "текст"},
        {"role": "user", "text": "/start"},
        {"role": "assistant", "text": "вопрос 1"}
    ]

    who_data_payload = [
        {"role": "system", "text": "кто ты?"},
        {"role": "user", "text": "/start"},
        {"role": "assistant", "text": "привет"}
    ]

    # --- SAVE ---
    about_id = repo.save_about(user_id, about_payload)
    about_data_id = repo.save_about_data(user_id, about_data_payload)
    test_data_id = repo.save_test_data(user_id, test_data_payload)
    who_data_id = repo.save_who_data(user_id, who_data_payload)

    print("Saved ids:", about_id, about_data_id, test_data_id, who_data_id)

    # --- GET LATEST ---
    got_about = repo.get_latest_about(user_id)
    got_about_data = repo.get_latest_about_data(user_id)
    got_test_data = repo.get_latest_test_data(user_id)
    got_who_data = repo.get_latest_who_data(user_id)

    assert got_about == about_payload, "about mismatch"
    assert got_about_data == about_data_payload, "about_data mismatch"
    assert got_test_data == test_data_payload, "test_data mismatch"
    assert got_who_data == who_data_payload, "who_data mismatch"

    print("Read-back assertions passed")

    # --- DELETE ---
    deleted = repo.delete_all_user_data(user_id)
    print("Deleted counts:", deleted)

    assert repo.get_latest_about(user_id) is None
    assert repo.get_latest_about_data(user_id) is None
    assert repo.get_latest_test_data(user_id) is None
    assert repo.get_latest_who_data(user_id) is None

    print("Delete assertions passed")

if __name__ == "__main__":
    main()
