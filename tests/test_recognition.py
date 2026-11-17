import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.inference import adaface
from gotham.inference.embedder import align112
from gotham.inference.models import ARCFACE, FUSED, _ArcFaceAdapter, make_detector, make_embedder
from gotham.inference.runtime import Session
from gotham.memory.linker import face_thumb
from gotham.memory.store import MemoryStore

DATA = Path(__file__).parent / "data"


@unittest.skipUnless(adaface.available(), "AdaFace model not installed")
class TestFusedRecogniser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = FaceAnalyzer()
        cls.emb = make_embedder()
        cls.lena = cv2.imread(str(DATA / "lena.jpg"))
        cls.sol = cv2.imread(str(DATA / "solvay.jpg"))
        cls.f = cls.an.analyze(cls.lena, AnalysisSettings()).faces[0]
        cls.sol_faces = cls.an.analyze(cls.sol, AnalysisSettings(small_faces=True, max_landmark_faces=64)).faces

    @classmethod
    def tearDownClass(cls):
        cls.an.close()

    def test_descriptor_is_two_unit_halves(self):
        self.assertEqual(self.emb.profile, FUSED)
        v, q = self.emb.embed_q(self.lena, self.f.kps)
        self.assertEqual(v.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)
        self.assertAlmostEqual(float(np.linalg.norm(v[:512])), 2 ** -0.5, places=4)
        self.assertAlmostEqual(float(np.linalg.norm(v[512:])), 2 ** -0.5, places=4)
        self.assertGreater(q, 0.5, "a clear face is recognisable")

    def test_same_person_vs_others(self):
        v, _ = self.emb.embed_q(self.lena, self.f.kps)
        dark = cv2.GaussianBlur((self.lena * 0.45).astype(np.uint8), (0, 0), 1.5)
        g = self.an.analyze(dark, AnalysisSettings()).faces[0]
        w, _ = self.emb.embed_q(dark, g.kps)
        self.assertGreater(float(v @ w), FUSED.match + 0.3, "same person, darker and blurred")
        others = self.emb.embed_q_many(self.sol, [x.kps for x in self.sol_faces])
        self.assertEqual(len(others), 29)
        self.assertLess(max(float(v @ o) for o, _ in others), FUSED.new, "29 different people")

    def test_batched_equals_single(self):
        kps = [x.kps for x in self.sol_faces[:10]]
        many = self.emb.embed_q_many(self.sol, kps)
        for k, (v, _) in zip(kps, many):
            single = self.emb.embed_crops([align112(self.sol, k)], flip=False)[0][0]
            self.assertGreater(float(v @ single), 0.999)


class TestMigration(unittest.TestCase):
    def test_profiles_survive_a_model_change(self):
        if not adaface.available():
            self.skipTest("AdaFace model not installed")
        an = FaceAnalyzer()
        lena = cv2.imread(str(DATA / "lena.jpg"))
        sol = cv2.imread(str(DATA / "solvay.jpg"))
        lf = an.analyze(lena, AnalysisSettings()).faces[0]
        sf = an.analyze(sol, AnalysisSettings(small_faces=True, max_landmark_faces=64)).faces
        an.close()
        arc = _ArcFaceAdapter()
        with tempfile.TemporaryDirectory() as d:
            old = MemoryStore(Path(d), profile=ARCFACE)
            people = [(lena, lf)] + [(sol, f) for f in sf[:6]]
            pids = []
            for img, f in people:
                v, q = arc.embed_q(img, f.kps)
                pids.append(old.create_person([(v, 0.0, 0.8, face_thumb(img, f.box, 72)),
                                               (v, 20.0, 0.7, face_thumb(img, f.box, 72))],
                                              face_thumb(img, f.box), 0.8, time.time()))
            old.set_notes(pids[0], "kept across the upgrade")
            old.add_sighting(pids[0], "live", "CAM", time.time() - 60, time.time(), 30, 0.9, None)
            old.close()

            st = MemoryStore(Path(d), profile=FUSED)
            self.assertEqual(st.migrate_from, ARCFACE.name)
            self.assertFalse(st.enabled, "no matching until the descriptors are re-made")
            self.assertEqual(st.match(np.ones((1, 1024), np.float32) / 32.0), [])
            r = st.migrate(make_embedder(), make_detector())
            self.assertTrue(st.enabled)
            self.assertEqual(r["profiles"], 7)
            self.assertEqual(st._mat.shape[1], 1024)
            self.assertEqual(st.person(pids[0]).notes, "kept across the upgrade")
            self.assertEqual(st.profile_stats(pids[0])["sightings"], 1)
            dark = cv2.GaussianBlur((lena * 0.45).astype(np.uint8), (0, 0), 1.5)
            g = FaceAnalyzer().analyze(dark, AnalysisSettings()).faces[0]
            v, _ = make_embedder().embed_q(dark, g.kps)
            ranked = st.match(v[None])
            self.assertEqual(ranked[0][0], pids[0])
            self.assertGreater(ranked[0][1], FUSED.match)
            self.assertLess(ranked[1][1], FUSED.new)
            path = st.path.parent
            st.close()
            again = MemoryStore(path, profile=FUSED)
            self.assertEqual(again.migrate_from, "")
            self.assertTrue(again.enabled)
            again.close()


class TestViewHygiene(unittest.TestCase):
    def test_non_finite_descriptors_are_never_stored(self):
        with tempfile.TemporaryDirectory() as d:
            st = MemoryStore(Path(d), profile=FUSED)
            bad = np.full(1024, np.nan, np.float32)
            good = np.ones(1024, np.float32) / 32.0
            pid = st.create_person([(bad, 0.0, 0.9), (good, 10.0, 0.9)], None, 0.5, time.time())
            self.assertEqual(len(st._pid), 1)
            self.assertFalse(st.learn(pid, bad, 50.0, 0.9))
            self.assertTrue(np.isfinite(st._mat).all())
            st.close()

    def test_audit_drops_unrecognisable_views_once(self):
        if not adaface.available():
            self.skipTest("AdaFace model not installed")
        an = FaceAnalyzer()
        lena = cv2.imread(str(DATA / "lena.jpg"))
        f = an.analyze(lena, AnalysisSettings()).faces[0]
        an.close()
        emb = make_embedder()
        v, _ = emb.embed_q(lena, f.kps)
        noise = np.random.default_rng(0).integers(0, 255, (72, 72, 3)).astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            st = MemoryStore(Path(d), profile=FUSED)
            good = st.create_person([(v, 0.0, 0.8, face_thumb(lena, f.box, 72)), (v, 20.0, 0.7, noise)], None, 0.5,
                                    time.time())
            only = st.create_person([(v, 0.0, 0.5, noise)], None, 0.5, time.time())
            st.db.execute("DELETE FROM meta WHERE key='views_audit'")
            st.close()
            (Path(d) / "memory.before-arcface_w600k_r50.db").write_bytes(b"")
            st = MemoryStore(Path(d), profile=FUSED)
            self.assertTrue(st.needs_audit)
            self.assertFalse(st.enabled, "recognition waits for the audit")
            r = st.audit_views(emb, make_detector())
            self.assertEqual(r["dropped"], 1)
            self.assertTrue(st.enabled)
            self.assertEqual(st.person(good).exemplars, 1)
            self.assertEqual(st.person(only).exemplars, 1, "a profile always keeps its best view")
            st.close()
            st = MemoryStore(Path(d), profile=FUSED)
            self.assertFalse(st.needs_audit, "once only")
            st.close()


class TestGpuRunSafety(unittest.TestCase):
    def test_two_models_from_two_threads(self):
        a = Session(Path(__file__).resolve().parent.parent / "models" / "w600k_r50_batch.onnx")
        b = Session(Path(__file__).resolve().parent.parent / "models" / "det_10g.onnx")
        if not (a.on_gpu and b.on_gpu):
            self.skipTest("no DirectML")
        self.assertIs(a._lock, b._lock)
        x = np.zeros((8, 3, 112, 112), np.float32)
        y = np.zeros((1, 3, 640, 640), np.float32)
        errors = []

        def loop(s, inp):
            try:
                for _ in range(60):
                    s.run(inp)
            except Exception as e:
                errors.append(e)
        th = [threading.Thread(target=loop, args=(a, x)), threading.Thread(target=loop, args=(b, y))]
        [t.start() for t in th]
        [t.join() for t in th]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
