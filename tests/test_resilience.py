import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.analysis import pipeline
from gotham.core.feed import Feed
from gotham.sources.capture import Kind, SourceSpec

VIDEO = Path(__file__).parent / "data" / "tracking_synth.mp4"


class TestResilience(unittest.TestCase):
    def test_feed_survives_intermittent_faults(self):
        real = pipeline.FaceAnalyzer.analyze
        calls = {"n": 0}

        def flaky(self, frame, s):
            calls["n"] += 1
            if calls["n"] % 4 == 0:
                raise RuntimeError("injected GPU hiccup")
            return real(self, frame, s)

        with mock.patch.object(pipeline.FaceAnalyzer, "analyze", flaky):
            feed = Feed(SourceSpec(Kind.FILE, str(VIDEO), loop=True))
            feed.start()
            feed.start_tracking()
            time.sleep(5)
            seq_mid = feed.latest().seq
            time.sleep(2)
            seq_end = feed.latest().seq
            feed.stop()
        print(f"\nframes analysed {seq_end}, faults survived {feed.faults}")
        self.assertGreater(feed.faults, 5)
        self.assertGreater(seq_end, seq_mid, "the feed kept producing results after faults")
        self.assertIn("fault", [e.kind for e in feed.events])


if __name__ == "__main__":
    unittest.main(verbosity=2)
