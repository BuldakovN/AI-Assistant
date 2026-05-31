"""
Интеграционные тесты для заглушки ``model/main.py`` (порт 8000, ``model/app_run.py``).

Монолит ``start_llm`` удалён: POST-эндпоинты отвечают **410 Gone** с полем ``detail``.
Рабочий диалог — сервис **core** (другой хост/порт).

Запуск:
    python -m unittest tests.test_api_endpoints
    или
    python -m pytest tests/test_api_endpoints.py
"""

import unittest
from datetime import datetime
from typing import Any, Dict

import requests

BASE_URL = "http://localhost:8000"
LEGACY_REMOVED = 410

TEST_USER_ID = "test_user_123"
TEST_PROMPT = "Привет, я хочу узнать о профессиях в IT"
TEST_PROFESSION_NAME = "Программист"


def _assert_gone(response: requests.Response) -> Dict[str, Any]:
    assert response.status_code == LEGACY_REMOVED, response.text
    data = response.json()
    assert "detail" in data
    return data


class TestAPIRootEndpoint(unittest.TestCase):
    def test_root_endpoint_status(self):
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API (http://localhost:8000). Запустите model/app_run.py")

    def test_root_endpoint_message(self):
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("message", data)
            self.assertIn("removed", data)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")


class TestAPIStartTalkEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.payload: Dict[str, Any] = {
            "user_id": self.user_id,
            "prompt": TEST_PROMPT,
            "parameters": {},
        }

    def test_start_talk_returns_gone(self):
        try:
            response = requests.post(
                f"{BASE_URL}/start_talk/",
                json=self.payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            _assert_gone(response)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")

    def test_start_talk_response_has_detail(self):
        try:
            response = requests.post(
                f"{BASE_URL}/start_talk/",
                json=self.payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            data = _assert_gone(response)
            self.assertIsInstance(data["detail"], str)
            self.assertIn("core", data["detail"].lower())
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")

    def test_start_talk_different_prompts(self):
        for prompt in (
            "Расскажи о профессии программиста",
            "Какие навыки нужны для работы в data science?",
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    "user_id": f"test_user_prompt_{datetime.now().timestamp()}",
                    "prompt": prompt,
                    "parameters": {},
                }
                try:
                    response = requests.post(
                        f"{BASE_URL}/start_talk/",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=5,
                    )
                    _assert_gone(response)
                except requests.exceptions.ConnectionError:
                    self.fail("Не удалось подключиться к API")


class TestAPIGetUserInfoEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.payload = {"user_id": self.user_id, "prompt": "", "parameters": {}}

    def test_get_user_info_gone(self):
        try:
            response = requests.post(
                f"{BASE_URL}/get_user_info/",
                json=self.payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            _assert_gone(response)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")


class TestAPIGetProfessionInfoEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.payload = {"user_id": self.user_id, "profession_name": TEST_PROFESSION_NAME}

    def test_get_profession_info_gone(self):
        try:
            response = requests.post(
                f"{BASE_URL}/get_profession_info/",
                json=self.payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            _assert_gone(response)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")

    def test_get_profession_info_different_professions(self):
        for profession in ("Data Scientist", "Веб-разработчик"):
            with self.subTest(profession=profession):
                payload = {
                    "user_id": f"test_user_{datetime.now().timestamp()}",
                    "profession_name": profession,
                }
                try:
                    response = requests.post(
                        f"{BASE_URL}/get_profession_info/",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=5,
                    )
                    _assert_gone(response)
                except requests.exceptions.ConnectionError:
                    self.fail("Не удалось подключиться к API")


class TestAPIDocumentation(unittest.TestCase):
    def test_swagger_ui_available(self):
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=5)
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")

    def test_redoc_available(self):
        try:
            response = requests.get(f"{BASE_URL}/redoc", timeout=5)
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")

    def test_openapi_schema_available(self):
        try:
            response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
            self.assertEqual(response.status_code, 200)
            schema = response.json()
            self.assertIsInstance(schema, dict)
            self.assertIn("openapi", schema)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")


class TestAPIComprehensive(unittest.TestCase):
    def test_legacy_posts_are_gone(self):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=5)
            self.assertEqual(r.status_code, 200)

            uid = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            for path, body in (
                ("/start_talk/", {"user_id": uid, "prompt": "hi", "parameters": {}}),
                ("/get_user_info/", {"user_id": uid, "prompt": "", "parameters": {}}),
                ("/get_profession_info/", {"user_id": uid, "profession_name": "X"}),
                ("/get_profession_roadmap/", {"user_id": uid, "profession_name": "X"}),
                ("/clean_history/", {"user_id": uid, "prompt": "", "parameters": {}}),
                ("/finish_test_early/", {"user_id": uid, "prompt": "", "parameters": {}}),
            ):
                resp = requests.post(f"{BASE_URL}{path}", json=body, timeout=5)
                _assert_gone(resp)
        except requests.exceptions.ConnectionError:
            self.fail("Не удалось подключиться к API")


if __name__ == "__main__":
    unittest.main(verbosity=2)
