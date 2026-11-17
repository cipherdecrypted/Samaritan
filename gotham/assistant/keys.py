from __future__ import annotations

import json
import os
from pathlib import Path

from ..memory import secure


def _path() -> Path:
    d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Gotham"
    d.mkdir(parents=True, exist_ok=True)
    return d / "keys.bin"


def load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(secure.unprotect(p.read_bytes()).decode())
    except Exception:
        return {}


def save(name: str, value: str | None):
    data = load()
    if value:
        data[name] = value
    else:
        data.pop(name, None)
    _path().write_bytes(secure.protect(json.dumps(data).encode()))


def get(name: str, env: str) -> str | None:
    return os.environ.get(env) or load().get(name)


def mask(text: str) -> str:
    t = text.strip()
    if t.lower().startswith(":key"):
        return ":key ••••••••"
    return t
