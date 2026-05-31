import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()
# Phoenix auto-instrument (LangChain) должен включиться до импорта llm_service / адаптера.
from model.phoenix_tracing import setup_phoenix_if_enabled

setup_phoenix_if_enabled()

if __name__ == "__main__":
    uvicorn.run("model.llm_service:app", reload=False, loop="asyncio", port=8001, host="0.0.0.0")
