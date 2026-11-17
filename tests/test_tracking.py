import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.inference.models import make_embedder
from gotham.tracking.assignment import linear_sum_assignment
from gotham.tracking.tracker import FaceTracker, State
from tests.synth import tracking_sequence


class TestAssignment(unittest.TestCase):
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for shape in [(3, 3), (2, 5), (5, 2), (4, 4), (1, 1), (6, 3)]:
            C = rng.random(shape)
            pairs = linear_sum_assignment(C)
            got = sum(C[i, j] for i, j in pairs)
            n, m = shape
            best = min(
                sum(C[i, p[i]] for i in range(n)) if n <= m else sum(C[p[j], j] for j in range(m))
                for p in (itertools.permutations(range(m), n) if n <= m else itertools.permutations(range(n), m))
            )
            self.assertAlmostEqual(got, best, places=9)
            self.assertEqual(len(pairs), min(n, m))


class TestTrackerSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = FaceAnalyzer()
        cls.emb = make_embedder()
        cls.settings = AnalysisSettings(landmarks=True)

    def run_sequence(self):
        tr = FaceTracker()
        tr.cfg.reacquire_similarity = self.emb.profile.reacquire
        tr.start(0.0)
        log = []
        for t, frame, gt in tracking_sequence():
            res = self.an.analyze(frame, self.settings)
            tracks = tr.update(res.faces, t, embed=lambda f, fr=frame: self.emb.embed(fr, f.kps))
            ids = {}
            for k in ("A", "B"):
                if k in gt and not gt.get(f"{k}_occluded"):
                    gx, gy = gt[k]
                    best = None
                    for track in tracks:
                        if track.id and track.visible:
                            c = track.face.centre
                            d = np.hypot(c[0] - (gx + 100), c[1] - (gy + 110))
                            if d < 160 and (best is None or d < best[0]):
                                best = (d, track.id)
                    ids[k] = best[1] if best else None
            log.append((t, gt, ids, {x.id: x.state for x in tracks if x.id}))
        return tr, log

    def test_identity_persistence(self):
        tr, log = self.run_sequence()
        a_ids = [ids.get("A") for _, gt, ids, _ in log if "A" in gt and not gt.get("A_occluded")]
        b_ids = [ids.get("B") for _, gt, ids, _ in log]
        a_ids = [i for i in a_ids[5:] if i is not None]
        b_ids = [i for i in b_ids[5:] if i is not None]
        print("\nA ids:", sorted(set(a_ids)), "B ids:", sorted(set(b_ids)),
              "subjects:", tr.total_subjects, "reacq:", tr.total_reacquisitions)
        self.assertGreater(len(a_ids), 100)
        self.assertGreater(len(b_ids), 150)
        self.assertEqual(len(set(a_ids)), 1, a_ids)
        self.assertEqual(len(set(b_ids)), 1, b_ids)
        self.assertNotEqual(a_ids[0], b_ids[0])
        self.assertEqual(tr.total_subjects, 2)
        self.assertGreaterEqual(tr.total_reacquisitions, 1)
        _, _, _, states = log[65]
        self.assertIn(State.COASTING, states.values())

    def test_session_reset_clears_everything(self):
        tr, _ = self.run_sequence()
        self.assertTrue(tr.tracks)
        tr.stop()
        self.assertEqual(tr.tracks, [])
        self.assertFalse(tr.active)
        tr.start(0.0)
        self.assertEqual(tr.total_subjects, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
