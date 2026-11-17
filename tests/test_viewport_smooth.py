import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from gotham.analysis.pipeline import AnalysisSettings, FaceAnalyzer
from gotham.core.feed import FeedResult, TrackView
from gotham.core.hub import Hub
from gotham.sources.capture import Status
from gotham.ui import assets
from gotham.ui.overlay import Layers
from gotham.ui.viewport import FeedView

DATA = Path(__file__).parent / "data"


class FakeCapture:
    def __init__(self):
        self.frame = None
        self.paused = False

    def latest(self):
        return self.frame


class TestSmoothVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        assets.load_fonts()
        img = cv2.imread(str(DATA / "lena.jpg"))
        frame = np.full((720, 1280, 3), 60, np.uint8)
        frame[100:100 + img.shape[0], 300:300 + img.shape[1]] = img[:620, :980] if img.shape[0] > 620 else img
        an = FaceAnalyzer()
        cls.face = an.analyze(frame, AnalysisSettings()).faces[0]
        an.close()
        cls.frame = frame

    def make_view(self, smooth: bool, velocity=(0.0, 0.0)):
        t0 = time.monotonic() - 0.2
        tv = TrackView(1, "S01", "active", self.face.box.copy(), self.face, True,
                       tuple((t0 - 0.05 * k, *self.face.centre) for k in range(5, 0, -1)),
                       np.array(velocity, np.float32), t0 - 1, t0, 30, 0, False)
        result = FeedResult(1, self.frame, t0, [self.face], [tv], {}, 15.0, 40.0)
        cap = FakeCapture()
        feed = NS(id=1, name="FEED-01", spec=NS(label="TEST"), provider="DmlExecutionProvider", is_file=False,
                  tracking=True, tracker=NS(cfg=NS(trail_seconds=6.0), started_at=t0 - 5),
                  stats=NS(fps=30.0, width=1280, height=720, position=0, status=Status.LIVE, reconnects=0, error=""),
                  capture=cap, latest=lambda: result)
        L = Layers()
        L.smooth, L.scan, L.grid, L.telemetry = smooth, False, False, False
        view = FeedView(feed, L, Hub())
        view.resize(1280, 720)
        return view, cap, result

    def render(self, view):
        img = QImage(1280, 720, QImage.Format_RGB32)
        view.render(img)
        return np.frombuffer(img.constBits(), np.uint8).reshape(720, 1280, 4)[..., :3].astype(int).copy()

    def test_sprites_match_direct_drawing(self):
        direct, _, _ = self.make_view(False)
        smooth, _, _ = self.make_view(True)
        a, b = self.render(direct), self.render(smooth)
        self.assertTrue(smooth._sprites, "overlays were cached as sprites")
        diff = np.abs(a - b).max(axis=2) > 48
        diff[:80] = False
        self.assertLess(int(diff.sum()), 400, "sprite rendering looks the same as direct drawing")

    def test_live_frame_shown_and_overlay_follows_motion(self):
        view, cap, result = self.make_view(True, velocity=(400.0, 0.0))
        before = self.render(view)
        newer = self.frame.copy()
        newer[:, :, 2] = 255
        cap.frame = NS(image=newer, t=result.t + 0.1, seq=99)
        after = self.render(view)
        self.assertGreater(after[700, 640, 2], 200)
        self.assertLess(before[700, 640, 2], 120)
        shift = view._shifts(result, view._xf, 0.1)[1]
        self.assertAlmostEqual(shift.x(), 40.0, places=3)
        self.assertAlmostEqual(shift.y(), 0.0, places=3)

    def test_shift_is_capped(self):
        view, _, result = self.make_view(True, velocity=(400.0, 0.0))
        self.render(view)
        self.assertAlmostEqual(view._shifts(result, view._xf, 2.0)[1].x(), 60.0, places=3)


if __name__ == "__main__":
    unittest.main()
