from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("GOTHAM_OR_MODEL", "anthropic/claude-haiku-5.5")


class OpenRouterError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def tools_openai(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def post(key: str, body: dict, timeout: float = 90) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "Gotham"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        msg = e.reason
        try:
            msg = json.loads(e.read().decode()).get("error", {}).get("message", msg)
        except Exception:
            pass
        raise OpenRouterError(e.code, str(msg)) from None
    except urllib.error.URLError as e:
        raise OpenRouterError(0, f"connection failed: {e.reason}") from None
    if "error" in data and not data.get("choices"):
        err = data["error"]
        raise OpenRouterError(int(err.get("code", 500)) if str(err.get("code", "")).isdigit() else 500,
                              str(err.get("message", err)))
    return data


def trim_history(msgs: list[dict], keep_turns: int = 10) -> list[dict]:
    starts = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    return msgs[starts[-keep_turns]:] if len(starts) > keep_turns else msgs
