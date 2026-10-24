"""Settings from the repo-root .env (and an optional backend/.env), then the process environment."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parent
load_dotenv(BACKEND.parent / ".env")
load_dotenv(BACKEND / ".env")


def vlm_disabled() -> bool:
    """VLM_DISABLED=1 skips LM Studio entirely (tests, machines without the model); descriptions use the fallback."""
    return os.environ.get("VLM_DISABLED") == "1"
