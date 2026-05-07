#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

_service_dir = Path(__file__).resolve().parent
_repo_root = _service_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_service_dir))

from app import app  # noqa: E402


if __name__ == "__main__":
    host = os.getenv("AGENT_REST_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_REST_PORT", "8040"))
    uvicorn.run(app, host=host, port=port)
