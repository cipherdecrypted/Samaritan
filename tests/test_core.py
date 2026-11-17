import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.assistant.observer import TOOLS, Observer
from gotham.core.feed import Feed
from gotham.core.hub import Hub
from gotham.sources.capture import Kind, SourceSpec, Status
from gotham.ui.overlay import LAYER_HELP, Layers

VIDEO = Path(__file__).parent / "data" / "tracking_synth.mp4"


class TestCorePipeline(unittest.TestCase):
    def test_file_feed_end_to_end(self):
        import tempfile
        from gotham.memory.store import MemoryStore
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hub = Hub()
        hub.memory = MemoryStore(Path(tmp.name))
        self.addCleanup(hub.memory.close)
        hub.display, hub.display_help = Layers(), LAYER_HELP
        feed = Feed(SourceSpec(Kind.FILE, str(VIDEO), loop=False))
        hub.add_feed(feed)
        w_new = hub.add_watch("new_person")
        w_count = hub.add_watch("count_at_least", feed="FEED-%02d" % feed.id, value=2, note="two in view")
        w_re = hub.add_watch("subject_reacquired")
        feed.start()
        feed.start_tracking()
        t0 = time.time()
        while feed.stats.status != Status.ENDED and time.time() - t0 < 30:
            hub.tick()
            time.sleep(0.05)
        time.sleep(0.3)
        hub.tick()
        feed.stop()

        self.assertEqual(feed.stats.status.value, "stopped")
        self.assertGreaterEqual(feed.stats.frames, 200)
        kinds = [e.kind for e in hub.events]
        print("\nevents:", kinds)
        self.assertIn("session_start", kinds)
        self.assertGreaterEqual(kinds.count("acquired"), 2)
        self.assertIn("occluded", kinds)
        self.assertIn("reacquired", kinds)
        alerts = hub.recent_alerts()
        print("alerts:", alerts)
        self.assertTrue(any(a["watch_id"] == w_count.id for a in alerts))
        self.assertTrue(any(a["watch_id"] == w_re.id for a in alerts))
        self.assertEqual(sum(a["watch_id"] == w_new.id for a in alerts), 2)
        self.assertEqual(len(hub.people()["people"]), 2)
        self.assertEqual(len(hub.person_detail("P-0001")["history"]), 1)
        self.assertIn("people", hub.people_seen(600))

        obs = Observer(hub, load_saved=False)
        args = {
            "get_overview": {}, "get_feed_status": {"feed": None},
            "list_subjects": {"feed": None, "include_departed": True},
            "get_subject": {"subject": "S01", "feed": None}, "get_faces": {"feed": None},
            "get_events": {"seconds": None, "feed": None, "kinds": None}, "get_alerts": {},
            "create_watch": {"kind": "dwell_exceeds", "feed": None, "subject": None, "person": None, "value": 10,
                             "note": "", "once": True},
            "list_people": {"query": None, "limit": None}, "get_person": {"person": "P-0001"},
            "people_seen": {"seconds": None}, "label_person": {"person": "P-0001", "label": "Visitor"},
            "get_activity": {"feed": None, "minutes": None}, "take_snapshot": {"feed": None},
            "list_watches": {}, "cancel_watch": {"watch_id": w_count.id},
            "set_tracking": {"feed": "FEED-%02d" % feed.id, "active": False},
            "focus_subject": {"subject": None, "feed": None},
            "get_display": {}, "set_overlay": {"layer": "web", "enabled": True},
            "set_overlay_style": {"style": "machine"},
            "set_analysis": {"feed": None, "small_faces": True, "landmarks": None, "threshold": None},
            "get_time": {},
        }
        self.assertEqual(set(args), {t["name"] for t in TOOLS})
        for t in TOOLS:
            schema = t["input_schema"]
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            self.assertFalse(schema["additionalProperties"])
            out = obs.execute(t["name"], args[t["name"]])
            json.dumps(out, default=str)
        self.assertTrue(hub.display.web)
        self.assertEqual(hub.display.style, "machine")
        self.assertTrue(feed.settings.small_faces)
        print("subjects incl. departed:", json.dumps(hub.subjects(None, True), default=str)[:400])


if __name__ == "__main__":
    unittest.main(verbosity=2)
