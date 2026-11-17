import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.inference.models import make_embedder
from gotham.tracking.tracker import FaceTracker
from tests.synth import crossing_sequence, meet_and_return_sequence


class TestCrossing(unittest.TestCase):
    def run_cross(self, weight, seq=crossing_sequence):
        an, emb = FaceAnalyzer(), make_embedder()
        tr = FaceTracker()
        tr.cfg.appearance_weight = weight
        tr.start(0.0)
        owners = {"A": [], "B": []}
        for t, frame, gt in seq():
            tracks = tr.update(an.analyze(frame, AnalysisSettings()).faces, t,
                               embed=lambda f, fr=frame: emb.embed(fr, f.kps))
            for k, (gx, gy) in gt.items():
                best = None
                for x in tracks:
                    if x.id and x.misses == 0:
                        c = x.face.centre
                        d = np.hypot(c[0] - (gx + 100), c[1] - (gy + 110))
                        if d < 90 and (best is None or d < best[0]):
                            best = (d, x.id)
                if best:
                    owners[k].append(best[1])
        return owners

    def test_ids_survive_crossing(self):
        owners = self.run_cross(0.35)
        a_before, a_after = owners["A"][:10], owners["A"][-10:]
        b_before, b_after = owners["B"][:10], owners["B"][-10:]
        print(f"\nA: {sorted(set(owners['A']))} B: {sorted(set(owners['B']))}")
        self.assertEqual(set(a_before), set(a_after), "A kept its ID through the crossing")
        self.assertEqual(set(b_before), set(b_after), "B kept its ID through the crossing")
        self.assertNotEqual(a_before[0], b_before[0])

    def test_ids_survive_meet_and_turn_back(self):
        owners = self.run_cross(0.35, meet_and_return_sequence)
        self.assertEqual(set(owners["A"][:15]), set(owners["A"][-15:]))
        self.assertEqual(set(owners["B"][:15]), set(owners["B"][-15:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
