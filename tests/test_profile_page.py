import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis import geometry as geo
from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.core.feed import FeedEvent
from gotham.core.hub import Hub
from gotham.inference.models import SCRFD, recogniser_profile
from gotham.memory.linker import MemoryLinker
from gotham.memory.store import MemoryStore

DATA = Path(__file__).parent / "data"
RNG = np.random.default_rng(7)
DIM = recogniser_profile().dim


def unit(v):
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def near(base, sim):
    r = RNG.normal(size=base.shape)
    r -= (r @ base) * base
    r = r / np.linalg.norm(r)
    return unit(sim * base + np.sqrt(1 - sim ** 2) * r)


class FakeTrack:
    _uids = iter(range(1, 10_000))

    def __init__(self, tid, face):
        self.uid, self.id, self.face = next(self._uids), tid, face
        self.label, self.first_seen, self.last_seen, self.visible = f"S{tid:02d}", time.monotonic(), time.monotonic(), True
        self.person_id, self.person_score, self.memory_state = None, 0.0, ""


class TestProfilePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        an = FaceAnalyzer()
        lena = cv2.imread(str(DATA / "lena.jpg"))
        cls.lena = lena
        cls.face = an.analyze(lena, AnalysisSettings()).faces[0]
        an.close()
        assert cls.face.mesh is not None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name))
        self.base = unit(RNG.normal(size=DIM))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def person(self, base=None, n=3):
        base = self.base if base is None else base
        return self.store.create_person([(near(base, 0.9), -10 + 10 * k, 0.8) for k in range(n)],
                                        self.lena[100:300, 100:300], 0.5, time.time() - 600)

    def fake_face(self, mesh=True):
        f = self.face
        return NS(box=f.box, score=0.9, kps=f.kps, mesh=f.mesh if mesh else None, quality={"overall": 0.8},
                  pose=NS(yaw=2.0, pitch=3.0), recog_q=0.9)

    def test_notes_alert_mesh_roundtrip_and_persistence(self):
        pid = self.person()
        self.store.set_notes(pid, "  delivery driver, mornings  ")
        self.store.set_alert(pid, True)
        canon = geo.canonical_mesh(self.face.mesh)
        self.assertTrue(self.store.update_mesh(pid, canon, 0.6))
        self.assertFalse(self.store.update_mesh(pid, canon, 0.55), "only a better mesh replaces the stored one")
        path = self.store.path.parent
        self.store.close()
        self.store = MemoryStore(path)
        p = self.store.person(pid)
        self.assertEqual(p.notes, "delivery driver, mornings")
        self.assertTrue(p.alert)
        self.assertEqual(self.store.alert_ids(), {pid})
        m = self.store.mesh(pid)
        self.assertEqual(m.shape, (478, 3))
        self.assertLess(np.abs(m - canon).max(), 0.01)
        r = (m[geo.R_EYE["outer"], :2] + m[geo.R_EYE["inner"], :2]) / 2
        l_ = (m[geo.L_EYE["outer"], :2] + m[geo.L_EYE["inner"], :2]) / 2
        self.assertAlmostEqual(float(np.linalg.norm(l_ - r)), 1.0, places=2)
        self.assertAlmostEqual(float(l_[1] - r[1]), 0.0, places=2)
        self.assertLess(float(np.abs((l_ + r) / 2).max()), 0.01)

    def test_candidates_confirm_reject_and_merge(self):
        a, b = self.person(), self.person(unit(RNG.normal(size=DIM)))
        c1 = self.store.add_candidate(a, "CAM", time.time() - 50, 0.38, "S04", self.lena[:96, :96])
        c2 = self.store.add_candidate(a, "CAM", time.time() - 40, 0.36, "S05", None, other=b)
        c3 = self.store.add_candidate(a, "CAM", time.time() - 30, 0.37, "S06", None)
        self.assertEqual([c["id"] for c in self.store.candidates(a)], [c1, c3, c2], "most convincing first")
        n0 = self.store.profile_stats(a)["sightings"]
        self.assertEqual(self.store.resolve_candidate(c1, True), "confirmed")
        self.assertEqual(self.store.profile_stats(a)["sightings"], n0 + 1, "confirmed -> a sighting")
        self.assertEqual(self.store.resolve_candidate(c3, False), "rejected")
        self.assertEqual([c["id"] for c in self.store.candidates(a)], [c2])
        self.assertEqual(self.store.resolve_candidate(c2, True), "merged", "it became its own profile: merge")
        self.assertIsNone(self.store.person(b))
        self.assertEqual(self.store.candidates(a), [])

    def test_merge_keeps_notes_alert_mesh_and_moves_candidates(self):
        a, b = self.person(), self.person(unit(RNG.normal(size=DIM)))
        self.store.set_notes(a, "first")
        self.store.set_notes(b, "second")
        self.store.set_alert(b, True)
        self.store.update_mesh(b, geo.canonical_mesh(self.face.mesh), 0.7)
        c = self.store.add_candidate(b, "CAM", time.time(), 0.37, "S01", None)
        self.store.merge(a, b)
        p = self.store.person(a)
        self.assertEqual(p.notes, "first\nsecond")
        self.assertTrue(p.alert)
        self.assertEqual(self.store.alert_ids(), {a})
        self.assertIsNotNone(self.store.mesh(a))
        self.assertEqual([x["id"] for x in self.store.candidates(a)], [c])
        self.store.forget(a)
        self.assertEqual(self.store.alert_ids(), set())
        self.assertIsNone(self.store.candidate(c), "forgetting a profile removes its possible sightings")

    def test_profile_stats(self):
        pid = self.person()
        t0 = time.time() - 3 * 86400
        self.store.add_sighting(pid, "live", "FEED-A", t0, t0 + 120, 300, 0.71, None)
        self.store.add_sighting(pid, "live", "FEED-A", t0 + 1000, t0 + 1060, 100, 0.55, None)
        self.store.add_sighting(pid, "live", "FEED-B", time.time() - 30, time.time() - 10, 40, 0.83, None)
        st = self.store.profile_stats(pid)
        self.assertEqual(st["sightings"], 3)
        self.assertAlmostEqual(st["total_seconds"], 200, delta=1)
        self.assertEqual([s["source"] for s in st["sources"]], ["FEED-A", "FEED-B"])
        self.assertAlmostEqual(sum(st["hours"]), 200, delta=2, msg="time spread over hours of the day")
        self.assertEqual(sum(st["days"]), 3)
        self.assertEqual(st["days"][-1], 1, "today")
        self.assertAlmostEqual(st["best_score"], 0.83)
        self.assertGreater(st["coherence"], 0.6, "views of one person agree")
        self.assertEqual(sum(st["bins"]), st["views"])

    def test_fragments_become_visits(self):
        pid = self.person()
        t = time.time() - 1000
        self.store.add_sighting(pid, "live", "CAM-A", t, t + 20, 50, 0.6, None)
        self.store.add_sighting(pid, "live", "CAM-A", t + 50, t + 80, 60, 0.8, self.lena[:64, :64])
        self.store.add_sighting(pid, "live", "CAM-B", t + 60, t + 70, 10, 0.7, None)
        self.store.add_sighting(pid, "live", "CAM-A", t + 300, t + 310, 10, 0.5, None)
        self.assertEqual(self.store.compact_visits(), 1)
        rows = sorted(self.store.sighting_rows(pid), key=lambda r: (r["source"], r["start"]))
        self.assertEqual([(r["source"], round(r["end"] - r["start"])) for r in rows],
                         [("CAM-A", 80), ("CAM-A", 10), ("CAM-B", 10)])
        self.assertEqual((rows[0]["frames"], rows[0]["score"]), (110, 0.8))
        self.assertGreater(rows[0]["tsize"], 0, "the visit keeps a photo")
        self.assertEqual(self.store.resume_visit(pid, "CAM-A", t + 330)["id"], rows[1]["id"])
        self.assertIsNone(self.store.resume_visit(pid, "CAM-A", t + 400))
        self.assertIsNone(self.store.resume_visit(pid, "CAM-C", t + 330))

    def test_linker_continues_a_visit_after_occlusion(self):
        pid = self.person()
        lk = MemoryLinker(self.store, "FEED-V", detector=SCRFD)
        lk.frame = self.lena
        for k in range(2):
            tr = FakeTrack(20 + k, self.fake_face())
            for _ in range(3):
                lk.on_sample(tr, near(self.base, 0.85))
            self.assertEqual(tr.person_id, pid)
            tr.last_seen = time.monotonic()
            lk.on_frame([tr])
            lk.on_end(tr)
        self.assertEqual(self.store.profile_stats(pid)["sightings"], 1, "one visit, not two fragments")

    def test_associates(self):
        a, b, c = self.person(), self.person(unit(RNG.normal(size=DIM))), self.person(unit(RNG.normal(size=DIM)))
        t = time.time() - 5000
        self.store.add_sighting(a, "live", "CAM", t, t + 100, 10, 0.7, None)
        self.store.add_sighting(b, "live", "CAM", t + 40, t + 160, 10, 0.7, None)
        self.store.add_sighting(c, "live", "CAM", t + 400, t + 500, 10, 0.7, None)
        self.store.add_sighting(c, "live", "DOOR", t + 10, t + 90, 10, 0.7, None)
        got = self.store.associates(a)
        self.assertEqual([(x["id"], x["visits"], round(x["seconds"])) for x in got], [(b, 1, 60)])

    def test_gallery_photos_sent_once(self):
        from gotham.ui.memory_panel import MemoryPanel
        pid = self.person()
        panel = MemoryPanel(self.store)
        first = panel.payload({})
        self.assertTrue(first["people"][0]["thumb"])
        self.store.add_sighting(pid, "live", "X", time.time(), time.time(), 1, 0.5, None)
        panel._built = 0
        second = panel.payload({})
        self.assertIsNot(second, first)
        self.assertIsNone(second["people"][0]["thumb"], "already on the page: not re-sent")
        panel.page_reloaded()
        self.assertTrue(panel.payload({})["people"][0]["thumb"], "a reloaded page gets every photo again")

    def test_stray_view_is_found_and_removed_at_open(self):
        other = unit(RNG.normal(size=DIM))
        pid = self.store.create_person([(near(self.base, 0.9), -10.0, 0.8), (near(self.base, 0.9), 0.0, 0.8),
                                        (near(self.base, 0.9), 10.0, 0.8), (near(other, 0.95), 20.0, 0.8)],
                                       None, 0.5, time.time())
        pair = self.store.create_person([(near(self.base, 0.9), 0.0, 0.8), (near(other, 0.9), 5.0, 0.8)],
                                        None, 0.5, time.time())
        rep = self.store.view_report(pid)
        self.assertEqual([v["stray"] for v in rep], [False, False, False, True])
        self.assertLess(rep[3]["agreement"], 0.15)
        self.assertFalse(any(v["stray"] for v in self.store.view_report(pair)))
        path = self.store.path.parent
        self.store.close()
        self.store = MemoryStore(path)
        self.assertEqual(self.store.stray_removed, 1)
        self.assertEqual(self.store.person(pid).exemplars, 3)
        self.assertEqual(self.store.person(pair).exemplars, 2, "an undecidable pair is left to the operator")
        last = self.store.view_report(pair)[0]["id"]
        self.assertTrue(self.store.remove_view(last))
        self.assertFalse(self.store.remove_view(self.store.view_report(pair)[0]["id"]), "a profile keeps one view")

    def test_one_stray_view_cannot_make_a_certain_merge(self):
        other = unit(RNG.normal(size=DIM))
        x = near(other, 0.97)
        a = self.store.create_person([(near(self.base, 0.9), 0.0, 0.8), (near(self.base, 0.9), 10.0, 0.8), (x, 20.0, 0.8)],
                                     None, 0.5, time.time())
        b = self.store.create_person([(near(other, 0.97), 0.0, 0.8), (near(other, 0.97), 10.0, 0.8)], None, 0.5, time.time())
        best = max(sc for p, q, sc in self.store.duplicate_pairs(limit=100) if {p, q} == {a, b})
        self.assertGreater(best, self.store.certain_duplicate, "the single best view pair looks certain...")
        self.assertLess(self.store.pair_score(a, b), self.store.certain_whole, "...but the profiles do not match as wholes")
        self.assertEqual(self.store.merge_certain_duplicates(), [])
        self.assertEqual(len(self.store.persons()), 2)

    def test_enrolment_ignores_a_swapped_minority(self):
        lk = MemoryLinker(self.store, "FEED-T", detector=SCRFD)
        lk.frame = self.lena
        tr = FakeTrack(30, self.fake_face())
        x, y = unit(RNG.normal(size=DIM)), unit(RNG.normal(size=DIM))
        for v in [near(x, 0.9), near(y, 0.9), near(x, 0.9), near(x, 0.9)]:
            lk.on_sample(tr, v)
        self.assertEqual(tr.memory_state, "new")
        m = self.store._mat[self.store._pid == tr.person_id]
        self.assertTrue(((m @ x) > 0.6).all(), "the profile is made of the majority face only")

    def test_merge_certain_duplicates(self):
        a, b, c = self.person(), self.person(), self.person()
        other = self.person(unit(RNG.normal(size=DIM)))
        done = self.store.merge_certain_duplicates()
        self.assertEqual(len(done), 2)
        self.assertEqual({p.id for p in self.store.persons()}, {a, other}, "a chain folds into the oldest")
        self.assertEqual(self.store.person(a).exemplars, 3, "trimmed to the per-angle budget (all frontal here)")

    def test_duplicate_review_cached_until_descriptors_change(self):
        a = self.person()
        b = self.person()
        pairs = self.store.duplicate_pairs()
        self.assertEqual([(x, y) for x, y, _ in pairs], [(a, b)])
        self.assertIs(self.store._dup_cache[1], self.store._dup_cache[1])
        self.store.add_sighting(a, "live", "X", time.time(), time.time(), 1, 0.9, None)
        self.assertEqual(self.store.duplicate_pairs(), pairs, "sightings do not invalidate the cache")
        v = self.store.emb_version
        self.store.learn(a, near(self.base, 0.9), 60.0, 0.9)
        self.assertGreater(self.store.emb_version, v)
        self.store.merge(a, b)
        self.assertEqual(self.store.duplicate_pairs(), [])

    def test_grouped_match_equals_bruteforce(self):
        ids = [self.person(unit(RNG.normal(size=DIM)), n=int(RNG.integers(1, 6))) for _ in range(40)]
        self.store.forget(ids[3])
        self.store.merge(ids[5], ids[9])
        q = np.stack([unit(RNG.normal(size=DIM)) for _ in range(7)])
        w = RNG.uniform(0.2, 1.0, 7)
        got = dict(self.store.match(q, weights=w))
        for pid in set(ids) - {ids[3], ids[9]}:
            m = self.store._mat[self.store._pid == pid]
            want = float(((q @ m.T).max(1) * (w / w.sum())).sum())
            self.assertAlmostEqual(got[pid], want, places=5)
        sim = self.store.similar(ids[0], 3)
        self.assertEqual(len(sim), 3)
        self.assertNotIn(ids[0], [p for p, _ in sim])

    def test_near_miss_becomes_possible_sighting_then_merges(self):
        pid = self.person()
        lk = MemoryLinker(self.store, "FEED-T", detector=SCRFD)
        lk.frame = self.lena
        events = []
        lk.emit = lambda kind, tr, text, person: events.append((kind, person))
        tr = FakeTrack(4, self.fake_face())
        target = self.store.th.match - 0.035
        lookalike = near(self.base, target / (0.9 * 0.92))
        for _ in range(MemoryLinker.FORCE_NEW_AFTER):
            lk.on_sample(tr, near(lookalike, 0.92))
        self.assertIn(("possible_match", pid), events)
        self.assertEqual(tr.memory_state, "new", "never reached the match line: its own profile in the end")
        cands = self.store.candidates(pid)
        self.assertEqual(len(cands), 1, "one possible sighting per track, updated in place")
        self.assertEqual(cands[0]["other_id"], tr.person_id)
        self.assertIsNotNone(self.store.mesh(tr.person_id), "frontal mesh stored with the new profile")
        self.assertEqual(self.store.resolve_candidate(cands[0]["id"], True), "merged")
        self.assertEqual(len(self.store.persons()), 1)

    def test_possible_sighting_dropped_when_later_recognised(self):
        pid = self.person()
        lk = MemoryLinker(self.store, "FEED-T", detector=SCRFD)
        lk.frame = self.lena
        tr = FakeTrack(5, self.fake_face())
        for _ in range(2):
            lk.on_sample(tr, near(self.base, self.store.th.match - 0.03))
        self.assertEqual(len(self.store.candidates(pid)), 1)
        for _ in range(4):
            lk.on_sample(tr, near(self.base, 0.85))
        self.assertEqual(tr.memory_state, "known")
        self.assertEqual(tr.person_id, pid)
        self.assertEqual(self.store.candidates(pid), [], "it was them: the real sighting supersedes the near miss")

    def test_swapped_track_never_puts_another_face_on_the_profile(self):
        pid = self.person()
        lk = MemoryLinker(self.store, "FEED-T", detector=SCRFD)
        lk.frame = self.lena
        tr = FakeTrack(7, self.fake_face())
        for _ in range(3):
            lk.on_sample(tr, near(self.base, 0.85))
        self.assertEqual((tr.memory_state, tr.person_id), ("known", pid))
        m = lk._mem[tr.uid]
        own = m.thumb.copy()
        lk.frame = 255 - self.lena
        tr.face.quality = {"overall": 0.99}
        stranger = unit(RNG.normal(size=DIM))
        lk.on_sample(tr, near(stranger, 0.9))
        self.assertTrue(np.array_equal(m.thumb, own))
        for _ in range(5):
            lk.on_sample(tr, near(stranger, 0.9))
        self.assertNotIn(tr.person_id, (None, pid), "unlinked, then re-identified as whoever it is now")
        jpeg = self.store.sighting_thumb(self.store.sighting_rows(pid)[0]["id"])
        shot = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertLess(float(np.abs(shot.astype(int) - own.astype(int)).mean()), 12)
        self.assertGreater(float(np.abs(m.thumb.astype(int) - own.astype(int)).mean()), 40,
                           "the new identity got its own photo")

    def test_swapped_holder_does_not_lock_out_the_real_person(self):
        pid = self.person()
        lk = MemoryLinker(self.store, "FEED-T", detector=SCRFD)
        lk.frame = self.lena
        a = FakeTrack(8, self.fake_face())
        for _ in range(3):
            lk.on_sample(a, near(self.base, 0.85))
        self.assertEqual(a.person_id, pid)
        lk.on_sample(a, near(unit(RNG.normal(size=DIM)), 0.9))
        self.assertEqual(a.person_id, pid, "...not yet unlinked (needs several samples)")
        b = FakeTrack(9, self.fake_face())
        for _ in range(2):
            lk.on_sample(b, near(self.base, 0.85))
        self.assertEqual((b.memory_state, b.person_id), ("known", pid))
        self.assertEqual(len(self.store.persons()), 1, "no duplicate profile")

    def test_strangers_do_not_flood_possible_sightings(self):
        pid = self.person()
        lk = MemoryLinker(self.store, "FEED-T", detector=SCRFD)
        lk.frame = self.lena
        tr = FakeTrack(6, self.fake_face())
        for _ in range(4):
            lk.on_sample(tr, near(self.base, 0.15))
        self.assertEqual(self.store.candidates(pid), [])

    def test_watchlist_alert_and_person_detail(self):
        pid = self.person()
        self.store.set_alert(pid, True)
        self.store.set_notes(pid, "operator note")
        hub = Hub()
        hub.memory = self.store
        now = time.time()
        hub._on_event(None, FeedEvent(time.monotonic(), now, 2, "recognized", 3, "S03 recognised", pid))
        hub._on_event(None, FeedEvent(time.monotonic(), now + 5, 2, "recognized", 4, "S04 recognised", pid))
        _, alerts = hub.drain()
        wl = [a for a in alerts if a.title == "WATCHLIST"]
        self.assertEqual(len(wl), 1, "one alert, then throttled")
        self.assertIn("FEED-02", wl[0].text)
        d = hub.person_detail(pid)
        self.assertEqual(d["operator_notes"], "operator note")
        self.assertTrue(d["watchlist"])
        self.assertIn("possible_sightings", d)

    def test_page_payload(self):
        from gotham.ui.memory_panel import MemoryPanel
        pid = self.person()
        other = self.person(unit(RNG.normal(size=DIM)))
        self.store.update_mesh(pid, geo.canonical_mesh(self.face.mesh), 0.8)
        for k, sc in enumerate((0.61, 0.92, 0.75)):
            self.store.add_sighting(pid, "live", f"FEED-{k}", time.time() - 3600 * (k + 1), time.time() - 3600 * (k + 1) + 60,
                                    100, sc, self.lena[200:328, 200:328])
        self.store.add_candidate(pid, "FEED-9", time.time(), 0.37, "S09", self.lena[:64, :64], other=other)
        panel = MemoryPanel(self.store)
        panel.page = pid
        m = panel.payload({pid: ["FEED-01/S02"]})
        p = m["page"]
        self.assertEqual(p["id"], pid)
        self.assertEqual(p["live"], ["FEED-01/S02"])
        self.assertTrue(p["face"]["wire"].startswith("data:image/png;base64,"))
        self.assertTrue(p["face"]["signature"].startswith("data:image/png;base64,"))
        self.assertEqual(len(p["face"]["metrics"]), 11)
        self.assertEqual([x["score"] for x in p["best"]], ["0.92", "0.75", "0.61"], "highest confidence first")
        self.assertEqual(len(p["timeline"]), 3)
        self.assertTrue(all(x["thumb"] for x in p["timeline"]))
        self.assertEqual(len(p["candidates"]), 1)
        self.assertEqual(p["candidates"][0]["other_id"], other)
        self.assertEqual(len(p["hours"]), 24)
        self.assertAlmostEqual(max(p["hours"]), 1.0)
        self.assertEqual(len(p["days"]), 14)
        self.assertTrue(m["recent"])
        self.assertIs(panel.payload({pid: ["FEED-01/S02"]}), m, "unchanged: cached")
        self.store.add_sighting(pid, "live", "X", time.time(), time.time(), 1, 0.5, None)
        self.assertIs(panel.payload({pid: ["FEED-01/S02"]}), m, "store churn is throttled")
        panel.page = None
        self.assertIsNone(panel.payload({})["page"], "navigation is never throttled")


class TestPerformanceV4(unittest.TestCase):
    def test_tune_process(self):
        from gotham.perf import tune_process
        r = tune_process()
        if sys.platform == "win32":
            self.assertTrue(r["power_throttling_off"])
            self.assertEqual(r["blas_threads"], 4)

    def test_wireframe_render(self):
        from gotham.ui.memory_panel import render_signature, render_wireframe
        an = FaceAnalyzer()
        f = an.analyze(cv2.imread(str(DATA / "lena.jpg")), AnalysisSettings()).faces[0]
        an.close()
        img = render_wireframe(geo.canonical_mesh(f.mesh))
        self.assertEqual(img.shape, (360, 320, 3))
        self.assertGreater(int((img.max(axis=2) > 100).sum()), 3000, "the mesh is drawn")
        sig = render_signature(unit(RNG.normal(size=DIM)))
        self.assertEqual(sig.shape, (18, DIM, 3))


if __name__ == "__main__":
    unittest.main()
