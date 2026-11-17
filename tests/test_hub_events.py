import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.core.feed import Feed
from gotham.core.hub import Hub
from gotham.memory.store import MemoryStore
from gotham.sources.capture import Kind, SourceSpec, Status

VIDEO = Path(__file__).parent / "data" / "tracking_synth.mp4"


def run_until_end(hub, feed, timeout=40):
    t0 = time.time()
    while feed.stats.status != Status.ENDED and time.time() - t0 < timeout:
        hub.tick()
        time.sleep(0.05)
    hub.tick()


class TestTransit(unittest.TestCase):
    def test_person_walks_from_feed1_to_feed2(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hub = Hub()
        hub.memory = MemoryStore(Path(tmp.name))
        self.addCleanup(hub.memory.close)
        moves = hub.add_watch("person_moves")
        cam1 = Feed(SourceSpec(Kind.FILE, str(VIDEO), "CAM-NORTH", loop=False))
        hub.add_feed(cam1)
        cam1.start()
        cam1.start_tracking()
        run_until_end(hub, cam1)
        time.sleep(1.5)
        cam2 = Feed(SourceSpec(Kind.FILE, str(VIDEO), "CAM-SOUTH", loop=False))
        hub.add_feed(cam2)
        cam2.start()
        cam2.start_tracking()
        run_until_end(hub, cam2)
        cam1.stop()
        cam2.stop()
        transits = [e for e in hub.events if e.kind == "transit"]
        print("\n" + "\n".join(e.text for e in transits))
        self.assertEqual(len({e.person for e in transits}), 2, "both people seen moving between cameras")
        self.assertTrue(all(f"{cam1.name} -> {cam2.name}" in e.text for e in transits))
        self.assertTrue(any(a["watch_id"] == moves.id for a in hub.recent_alerts()))
        act = hub.activity_summary(None, 5)
        self.assertTrue(all(a.get("samples", 1) and a["peak"] >= 1 for a in act), act)
        self.assertEqual(len(hub.memory.persons()), 2, "no duplicate profiles across cameras")


class TestHygiene(unittest.TestCase):
    def test_duplicates_dismiss_prune(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        st = MemoryStore(Path(tmp.name))
        self.addCleanup(st.close)
        rng = np.random.default_rng(1)

        def u(v):
            return (v / np.linalg.norm(v)).astype(np.float32)
        base = u(rng.normal(size=st.th.dim))
        a = st.create_person([(base, 0, 0.9)], None, 0, time.time())
        b = st.create_person([(u(base + 0.35 * u(rng.normal(size=st.th.dim))), 0, 0.9)], None, 0, time.time())
        st.create_person([(u(rng.normal(size=st.th.dim)), 0, 0.9)], None, 0, time.time())
        pairs = st.duplicate_pairs()
        self.assertEqual([(x, y) for x, y, _ in pairs], [(a, b)])
        st.dismiss_pair(b, a)
        self.assertEqual(st.duplicate_pairs(), [])
        old_t = time.time() - 20 * 86400
        old = st.create_person([(u(rng.normal(size=st.th.dim)), 0, 0.9)], None, 0, old_t)
        st.add_sighting(old, "live", "FEED-01", old_t, old_t + 5, 30, 0.9, None)
        st.db.execute("UPDATE persons SET last_seen=? WHERE id=?", (old_t, old))
        st.rename(a, "Keeper")
        self.assertEqual(st.prunable(7), [old])
        self.assertEqual(st.prune(7), 1)
        self.assertEqual(len(st.persons()), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
