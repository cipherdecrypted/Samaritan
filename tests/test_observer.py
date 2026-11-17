import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.assistant.observer import Observer
from gotham.core.hub import Hub
from gotham.ui.overlay import LAYER_HELP, Layers


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.script.pop(0)


def tool_use(i, name, inp):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


class TestObserverLoop(unittest.TestCase):
    def test_tool_loop_and_reply(self):
        hub = Hub()
        hub.display, hub.display_help = Layers(), LAYER_HELP
        obs = Observer(hub, load_saved=False)
        fake = FakeMessages([
            NS(stop_reason="tool_use", content=[tool_use(1, "get_overview", {}),
                                                tool_use(2, "set_overlay", {"layer": "web", "enabled": True}),
                                                tool_use(3, "create_watch", {"kind": "subject_acquired", "feed": None,
                                                                             "subject": None, "value": None,
                                                                             "note": "tell me", "once": False})]),
            NS(stop_reason="end_turn", content=[NS(type="thinking", thinking=""),
                                                NS(type="text", text="FACE WEB ONLINE\nWatch armed for new subjects.")]),
        ])
        obs._client = NS(beta=NS(messages=fake))
        done = threading.Event()
        got = {}
        obs.on_reply = lambda h, b: (got.update(head=h, body=b), done.set())
        obs.on_error = lambda m: (got.update(error=m), done.set())
        obs.ask("show me the face web and tell me when someone new shows up")
        self.assertTrue(done.wait(10))
        self.assertNotIn("error", got)
        self.assertEqual(got["head"], "FACE WEB ONLINE")
        self.assertTrue(hub.display.web)
        self.assertEqual(len(hub.watches), 1)
        second = fake.calls[1]["messages"]
        results = second[-1]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["tu_1", "tu_2", "tu_3"])
        self.assertFalse(any(r.get("is_error") for r in results))
        req = fake.calls[0]
        self.assertEqual(req["model"], "claude-opus-5-5")
        self.assertEqual(req["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", req["betas"])
        self.assertEqual(req["output_config"], {"effort": "medium"})

    def test_bad_tool_input_reported_not_raised(self):
        hub = Hub()
        hub.display, hub.display_help = Layers(), LAYER_HELP
        obs = Observer(hub, load_saved=False)
        fake = FakeMessages([
            NS(stop_reason="tool_use", content=[tool_use(1, "create_watch", {"kind": "count_at_least", "feed": "FEED-09",
                                                                             "subject": None, "value": 2, "note": "",
                                                                             "once": False})]),
            NS(stop_reason="end_turn", content=[NS(type="text", text="NO SUCH FEED")]),
        ])
        obs._client = NS(beta=NS(messages=fake))
        done = threading.Event()
        obs.on_reply = lambda h, b: done.set()
        obs.ask("watch feed 9")
        self.assertTrue(done.wait(10))
        res = fake.calls[1]["messages"][-1]["content"][0]
        self.assertTrue(res["is_error"])
        self.assertIn("no such feed", res["content"])

    def test_auth_failure_rolls_back_history(self):
        import anthropic
        hub = Hub()
        obs = Observer(hub, api_key="sk-invalid", load_saved=False)
        done = threading.Event()
        got = {}
        obs.on_error = lambda m: (got.update(error=m), done.set())
        obs.on_reply = lambda h, b: done.set()

        class Boom:
            def create(self, **kw):
                raise anthropic.APIConnectionError(request=None)
        obs._client = NS(beta=NS(messages=Boom()))
        obs.ask("hello")
        self.assertTrue(done.wait(10))
        self.assertIn("CONNECTION", got.get("error", ""))
        self.assertEqual(obs.history, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
