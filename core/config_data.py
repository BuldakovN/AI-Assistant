"""Загрузка YAML-конфигов core (промпты, каталог тестов)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_core_dir = Path(__file__).resolve().parent
_repo_root = _core_dir.parent
_model_dir = _repo_root / "model"
if (_core_dir / "config.yaml").exists():
    _config_dir = Path(os.getenv("CORE_CONFIG_DIR", str(_core_dir)))
else:
    _config_dir = Path(os.getenv("CORE_CONFIG_DIR", str(_model_dir)))

config_path = _config_dir / "config.yaml"
prof_tests_path = _config_dir / "prof_tests.yaml"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


config = load_yaml(config_path)
prof_tests = load_yaml(prof_tests_path)

EDUCATION_JSON = Path(
    os.getenv("EDUCATION_JSON_PATH", str(_repo_root / "data" / "education" / "education_detailed.json"))
)

folder_id = os.getenv("YANDEX_CLOUD_FOLDER", "")
api_key = os.getenv("YANDEX_CLOUD_API_KEY", "")
web_search_api_url = os.getenv("WEB_SEARCH_API_URL", "http://localhost:1000")
