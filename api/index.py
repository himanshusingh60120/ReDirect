# api/index.py
"""
Vercel entry point.

Vercel's Python runtime looks for a module-level `app` and serves it as an ASGI
application, so this file only has to expose the one from app/main.py. The
sys.path line keeps the import working whatever directory the runtime starts in.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402,F401
