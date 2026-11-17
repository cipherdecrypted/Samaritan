import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.assistant import keys
from gotham.assistant import openrouter as orr
from gotham.assistant.observer import SYSTEM, Observer
from gotham.core.hub import Hub
from gotham.ui.overlay import LAYER_HELP, Layers


def call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Server:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, key, body, timeout=90):
        self.bodies.append(json.loads(json.dumps(body)))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def reply(content="", calls=None, cost=0.0004):
    return {"choices": [{"finish_reason": "tool_calls" if calls else "stop",
                         "message": {"role": "assistant", "content": content, "tool_calls": calls}}],
            "usage": {"cost": cost}}


class TestOpenRouter(unittest.TestCase):
    def make(self):
        hub = Hub()
        hub.display, hub.display_help = Layers(), LAYER_HELP
        obs = Observer(hub, load_saved=False)
        obs.or_key = "sk-or-test"
        return hub, obs

    def run_ask(self, obs, text):
        done, got = threading.Event(), {}
        obs.on_reply = lambda h, b: (got.update(head=h, body=b), done.set())
        obs.on_error = lambda m: (got.update(error=m), done.set())
        obs.ask(text)
        self.assertTrue(done.wait(10))
        return got

    def test_tool_loop(self):
        hub, obs = self.make()
        srv = Server([reply(calls=[call(1, "set_overlay", {"layer": "web", "enabled": True}),
                                   call(2, "get_overview", {})]),
                      reply("FACE WEB ONLINE\nNO FEEDS CONNECTED")])
        with mock.patch.object(orr, "post", srv):
            got = self.run_ask(obs, "show me the face web")
        self.assertEqual(got.get("head"), "FACE WEB ONLINE")
        self.assertTrue(hub.display.web)
        first, second = srv.bodies
        self.assertEqual(first["model"], "anthropic/claude-haiku-5.5")
        self.assertEqual(first["messages"][0]["content"][0]["text"], SYSTEM)
        self.assertEqual(first["messages"][0]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(first["reasoning"], {"effort": "low"})
        self.assertTrue(all(t["type"] == "function" for t in first["tools"]))
        tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["call_1", "call_2"])
        self.assertAlmostEqual(obs.spent, 0.0008)
        srv2 = Server([reply("ACKNOWLEDGED")])
        with mock.patch.object(orr, "post", srv2):
            self.run_ask(obs, "and now?")
        roles = [m["role"] for m in srv2.bodies[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "tool", "assistant", "user"])

    def test_errors_and_reasoning_fallback(self):
        _, obs = self.make()
        srv = Server([orr.OpenRouterError(402, "Insufficient credits")])
        with mock.patch.object(orr, "post", srv):
            got = self.run_ask(obs, "hi")
        self.assertTrue(got["error"].startswith("INSUFFICIENT CREDITS"))
        srv = Server([orr.OpenRouterError(400, "reasoning is not supported for this model"), reply("ONLINE")])
        with mock.patch.object(orr, "post", srv):
            got = self.run_ask(obs, "status")
        self.assertEqual(got["head"], "ONLINE")
        self.assertNotIn("reasoning", srv.bodies[1])

    def test_bad_tool_arguments_reported(self):
        _, obs = self.make()
        bad = {"id": "c1", "type": "function", "function": {"name": "get_subject", "arguments": "{not json"}}
        srv = Server([reply(calls=[bad]), reply("MALFORMED REQUEST")])
        with mock.patch.object(orr, "post", srv):
            self.run_ask(obs, "x")
        tool = [m for m in srv.bodies[1]["messages"] if m["role"] == "tool"][0]
        self.assertTrue(tool["content"].startswith("Error:"))

    def test_key_storage_encrypted_and_masked(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"LOCALAPPDATA": d}):
            keys.save("openrouter", "sk-or-v1-secretvalue")
            raw = (Path(d) / "Gotham" / "keys.bin").read_bytes()
            self.assertNotIn(b"secretvalue", raw)
            self.assertEqual(keys.load()["openrouter"], "sk-or-v1-secretvalue")
            self.assertEqual(keys.mask(":key sk-or-v1-secretvalue"), ":key ••••••••")
            hub = Hub()
            obs = Observer(hub)
            self.assertEqual(obs.provider, "openrouter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
