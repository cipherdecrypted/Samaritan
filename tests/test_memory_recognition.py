import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.inference.models import FUSED, SCRFD, make_embedder, recogniser_profile
from gotham.memory.linker import MemoryLinker
from gotham.memory.store import MemoryStore

DATA = Path(__file__).parent / "data"


class FakeTrack:
    _uids = iter(range(1, 10_000))

    def __init__(self, tid, face):
        self.uid, self.id, self.face = next(self._uids), tid, face
        self.label, self.first_seen, self.last_seen, self.visible = f"S{tid:02d}", time.monotonic(), time.monotonic(), True
        self.person_id, self.person_score, self.memory_state = None, 0.0, ""


class TestMemoryV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = FaceAnalyzer()
        cls.emb = make_embedder()
        cls.s = AnalysisSettings(small_faces=True, max_landmark_faces=64, min_landmark_size=16)
        sol = cv2.imread(str(DATA / "solvay.jpg"))
        cls.sol_faces = cls.an.analyze(sol, cls.s).faces
        cls.sol = sol

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_uses_best_models(self):
        self.assertEqual(self.an.profile.name, SCRFD.name)
        self.assertEqual(recogniser_profile(), FUSED, "both recognition models are installed: fused is used")
        self.assertEqual(self.emb.profile.name, FUSED.name)
        self.assertEqual(self.store.th.name, FUSED.name)
        self.assertEqual(len(self.sol_faces), 29)

    def feed_samples(self, linker, track, img, face, n):
        for _ in range(n):
            v, face.recog_q = self.emb.embed_q(img, face.kps)
            linker.on_sample(track, v)

    def test_fast_enrolment_and_recognition(self):
        lena = cv2.imread(str(DATA / "lena.jpg"))
        f = self.an.analyze(lena, self.s).faces[0]
        lk = MemoryLinker(self.store, "TEST", detector=SCRFD)
        lk.frame = lena
        tr = FakeTrack(1, f)
        self.feed_samples(lk, tr, lena, f, MemoryLinker.NEW_AFTER)
        self.assertEqual(tr.memory_state, "new", "a clear stranger is enrolled after 3 samples")
        self.assertEqual(len(self.store.persons()), 1)
        dark = cv2.GaussianBlur((lena * 0.45).astype(np.uint8), (0, 0), 1.5)
        g = self.an.analyze(dark, self.s).faces[0]
        lk2 = MemoryLinker(self.store, "TEST", detector=SCRFD)
        lk2.frame = dark
        tr2 = FakeTrack(2, g)
        self.feed_samples(lk2, tr2, dark, g, MemoryLinker.MIN_KNOWN)
        self.assertEqual(tr2.memory_state, "known")
        self.assertEqual(tr2.person_id, tr.person_id)
        self.assertEqual(len(self.store.persons()), 1)

    def test_29_people_29_profiles_and_no_cross_matches(self):
        lk = MemoryLinker(self.store, "TEST", detector=SCRFD)
        lk.frame = self.sol
        tracks = []
        for i, f in enumerate(self.sol_faces):
            t = FakeTrack(i + 1, f)
            self.feed_samples(lk, t, self.sol, f, MemoryLinker.NEW_AFTER)
            if t.memory_state == "uncertain":
                self.feed_samples(lk, t, self.sol, f, MemoryLinker.FORCE_NEW_AFTER - MemoryLinker.NEW_AFTER)
            tracks.append(t)
            lk.on_end(t)
        ids = [t.person_id for t in tracks]
        print("\nsolvay enrolment states:", sorted({t.memory_state for t in tracks}))
        self.assertEqual(len([i for i in ids if i]), 29)
        self.assertEqual(len(set(ids)), 29, "every person must get their own profile")
        top = max(s for p in self.store.persons() for _, s in self.store.similar(p.id, 1))
        print("highest look-alike score between different people:", round(top, 3))
        self.assertLess(top, self.store.th.match)

    def test_identity_swap_is_unlinked_not_learned(self):
        a, b = self.sol_faces[0], self.sol_faces[5]
        lk = MemoryLinker(self.store, "TEST", detector=SCRFD)
        lk.frame = self.sol
        tr = FakeTrack(1, a)
        self.feed_samples(lk, tr, self.sol, a, MemoryLinker.NEW_AFTER)
        pid = tr.person_id
        views_before = len(self.store.views(pid))
        events = []
        lk.emit = lambda kind, track, text, person: events.append(kind)
        tr.face = b
        self.feed_samples(lk, tr, self.sol, b, 6)
        self.assertIn("unlinked", events)
        self.assertNotEqual(tr.person_id, pid)
        self.assertEqual(len(self.store.views(pid)), views_before, "the wrong person must not be learned")

    def test_view_gallery_and_lookalikes(self):
        lena = cv2.imread(str(DATA / "lena.jpg"))
        lk = MemoryLinker(self.store, "TEST", detector=SCRFD)
        tr = None
        for k, im in enumerate([lena, cv2.flip(lena, 1)]):
            f = self.an.analyze(im, self.s).faces[0]
            lk.frame = im
            if tr is None:
                tr = FakeTrack(1, f)
            tr.face = f
            self.feed_samples(lk, tr, im, f, 3)
        views = self.store.views(tr.person_id)
        self.assertGreaterEqual(len(views), 2)
        self.assertTrue(all(v["thumb"] for v in views), "every learned view keeps a snapshot")
        yaws = sorted(v["yaw"] for v in views)
        self.assertLess(yaws[0], 0)
        self.assertGreater(yaws[-1], 0)

    def test_model_change_resets_memory(self):
        db = sqlite3.connect(self.store.path)
        db.execute("UPDATE meta SET value='sface_2021dec' WHERE key='embed_model'")
        db.commit()
        db.close()
        dim = self.store.th.dim
        self.store.create_person([(np.ones(dim, np.float32) / dim ** 0.5, 0, 0.9)], None, 0, time.time())
        self.store.close()
        again = MemoryStore(Path(self.tmp.name))
        self.assertIn("recognition model changed", again.reset_reason)
        self.assertEqual(again.persons(), [])
        self.store = again


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReentryAndSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = FaceAnalyzer()
        cls.emb = make_embedder()
        cls.lena = cv2.imread(str(DATA / "lena.jpg"))
        cls.face = cls.an.analyze(cls.lena, AnalysisSettings()).faces[0]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name))
        self.lk = MemoryLinker(self.store, "TEST", detector=SCRFD)
        self.lk.frame = self.lena

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def feed(self, tr, n):
        for _ in range(n):
            v, self.face.recog_q = self.emb.embed_q(self.lena, self.face.kps)
            self.lk.on_sample(tr, v)

    def test_walking_back_in_under_a_new_subject_id_is_recognised(self):
        first = FakeTrack(1, self.face)
        self.feed(first, 3)
        pid = first.person_id
        first.visible = False
        self.lk.on_frame([first])
        again = FakeTrack(3, self.face)
        self.feed(again, 3)
        self.assertEqual(again.memory_state, "known")
        self.assertEqual(again.person_id, pid)
        self.assertEqual(len(self.store.persons()), 1, "no duplicate profile for a returning person")

    def test_two_people_in_view_cannot_share_one_identity(self):
        first = FakeTrack(1, self.face)
        self.feed(first, 3)
        self.lk.on_frame([first])
        twin = FakeTrack(2, self.face)
        self.feed(twin, 3)
        self.assertNotEqual(twin.person_id, first.person_id)

    def test_split_profiles_do_not_block_recognition(self):
        v, _ = self.emb.embed_q(self.lena, self.face.kps)
        rng = np.random.default_rng(3)
        n = rng.normal(size=v.shape)
        dup = v + 0.25 * n / np.linalg.norm(n)
        a = self.store.create_person([(v, 0, 0.9)], None, 0, time.time())
        self.store.create_person([((dup / np.linalg.norm(dup)).astype(np.float32), 0, 0.9)], None, 0, time.time())
        tr = FakeTrack(1, self.face)
        self.feed(tr, 3)
        self.assertEqual(tr.memory_state, "known")
        self.assertEqual(tr.person_id, a)
