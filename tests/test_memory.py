import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.core.feed import Feed
from gotham.inference.models import make_embedder
from gotham.memory.store import MemoryStore
from gotham.sources.capture import Kind, SourceSpec, Status

DATA = Path(__file__).parent / "data"


def run_session(store, video=DATA / "tracking_synth.mp4"):
    feed = Feed(SourceSpec(Kind.FILE, str(video), loop=False))
    feed.memory = store
    feed.start()
    feed.start_tracking()
    t0 = time.time()
    while feed.stats.status != Status.ENDED and time.time() - t0 < 40:
        time.sleep(0.05)
    time.sleep(0.3)
    feed.stop()
    return [e for e in feed.events if e.kind in ("recognized", "new_person")]


class TestMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = FaceAnalyzer()
        cls.emb = make_embedder()
        cls.s = AnalysisSettings(threshold=None, small_faces=True, max_landmark_faces=64, min_landmark_size=16)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_remembers_people_across_sessions(self):
        ev1 = run_session(self.store)
        persons1 = self.store.persons()
        print("\nsession 1:", [(e.kind, e.text) for e in ev1])
        self.assertEqual(len(persons1), 2, [p.code for p in persons1])
        frames1 = {p.id: sum(h["frames"] for h in self.store.sightings(p.id)) for p in persons1}
        ev2 = run_session(self.store)
        print("session 2:", [(e.kind, e.text) for e in ev2])
        persons2 = self.store.persons()
        self.assertEqual(len(persons2), 2, "second session must not create new profiles")
        self.assertEqual(sum(e.kind == "recognized" for e in ev2), 2)
        for p in persons2:
            hist = self.store.sightings(p.id)
            self.assertEqual(len(hist), 1, p.code)
            self.assertGreater(hist[0]["frames"], frames1[p.id], "session 2 was added to the visit")
            self.assertTrue(all(h["end"] >= h["start"] and h["frames"] > 0 for h in hist))
            self.assertIsNotNone(self.store.thumb(p.id))
            print(p.code, "sightings", len(hist), "exemplars", p.exemplars, "pose bins", p.bins)

    def test_no_false_merges_among_29_people(self):
        img = cv2.imread(str(DATA / "solvay.jpg"))
        faces = self.an.analyze(img, self.s).faces
        self.assertEqual(len(faces), 29)
        embs = [self.emb.embed(img, f.kps) for f in faces]
        pids = [self.store.create_person([(e, f.pose.yaw, f.quality["overall"])], None, 0, time.time())
                for e, f in zip(embs, faces)]
        wrong = 0
        for e, pid in zip(embs, pids):
            ranked = self.store.match(e[None])
            self.assertEqual(ranked[0][0], pid)
            others = [s for p, s in ranked if p != pid]
            wrong += max(others) >= self.store.th.match
        print("\nsolvay: highest cross-person score",
              round(max(s for e, pid in zip(embs, pids) for p, s in self.store.match(e[None]) if p != pid), 3))
        self.assertEqual(wrong, 0)

    def test_matches_same_person_under_variation(self):
        lena = cv2.imread(str(DATA / "lena.jpg"))
        f = self.an.analyze(lena, self.s).faces[0]
        pid = self.store.create_person([(self.emb.embed(lena, f.kps), f.pose.yaw, 0.9)], None, 0, time.time())
        variants = {
            "mirror": cv2.flip(lena, 1),
            "rotated 25°": cv2.warpAffine(lena, cv2.getRotationMatrix2D((256, 256), 25, 1), (512, 512)),
            "dark": (lena * 0.35).astype(np.uint8),
            "blurred": cv2.GaussianBlur(lena, (0, 0), 2.5),
            "low-res": cv2.resize(cv2.resize(lena, (128, 128)), (512, 512)),
            "grey": cv2.cvtColor(cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        }
        for name, im in variants.items():
            g = max(self.an.analyze(im, self.s).faces, key=lambda x: x.box[3])
            ranked = self.store.match(self.emb.embed(im, g.kps)[None])
            self.assertEqual(ranked[0][0], pid, name)
            self.assertGreaterEqual(ranked[0][1], self.store.th.match, name)

    def test_management_and_encryption(self):
        e = np.random.default_rng(0).normal(size=self.store.th.dim).astype(np.float32)
        e /= np.linalg.norm(e)
        a = self.store.create_person([(e, 0, 0.9)], np.zeros((8, 8, 3), np.uint8), 0.9, time.time())
        b = self.store.create_person([(e, 20, 0.8)], None, 0, time.time())
        self.store.add_sighting(b, "live", "FEED-X", time.time() - 5, time.time(), 10, 0.7, None)
        self.store.rename(a, "Front door regular")
        self.store.merge(a, b)
        ps = self.store.persons()
        self.assertEqual(len(ps), 1)
        self.assertEqual(ps[0].label, "Front door regular")
        self.assertEqual(len(self.store.sightings(a)), 1)
        raw = Path(self.store.path).read_bytes()
        self.assertNotIn(e.tobytes()[:32], raw, "embeddings must not be stored in the clear")
        self.store.forget(a)
        self.assertEqual(self.store.persons(), [])
        self.assertEqual(self.store.match(e[None]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
