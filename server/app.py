"""
server/app.py — OpenEnv multi-mode entry point.
Re-exports the ShadowWatch FastAPI app from api/main.py.
"""

import sys
import os

# Ensure repo root is on the path so api/ and env/ are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app  # noqa: F401 — re-export for openenv

__all__ = ["app"]