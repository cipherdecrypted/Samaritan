import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gotham.sources.capture import Capture, Kind, SourceSpec, Status

VIDEO = Path(__file__).parent / "data" / "tracking_synth.mp4"


def _frames():
    cap = cv2.VideoCapture(str(VIDEO))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    return out


FRAMES = _frames()


class MJPEG(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        i = 0
        try:
            while not self.server.stopping:
                jpg = FRAMES[i % len(FRAMES)]
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                 str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                i += 1
                time.sleep(1 / 25)
        except (ConnectionError, OSError):
            pass


def serve(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), MJPEG)
    srv.stopping = False
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TestStream(unittest.TestCase):
    def test_mjpeg_stream_with_reconnect(self):
        port = free_port()
        srv = serve(port)
        cap = Capture(SourceSpec(Kind.STREAM, f"http://127.0.0.1:{port}/stream.mjpg"))
        cap.start()
        t0 = time.time()
        while cap.stats.frames < 30 and time.time() - t0 < 20:
            time.sleep(0.1)
        print(f"\nlive: status={cap.stats.status.value} frames={cap.stats.frames} fps={cap.stats.fps:.1f} "
              f"{cap.stats.width}x{cap.stats.height} backend={cap.stats.backend}")
        self.assertEqual(cap.stats.status, Status.LIVE)
        self.assertGreaterEqual(cap.stats.frames, 30)
        self.assertEqual((cap.stats.width, cap.stats.height), (1280, 720))

        srv.stopping = True
        srv.shutdown()
        srv.server_close()
        t0 = time.time()
        while cap.stats.status == Status.LIVE and time.time() - t0 < 20:
            time.sleep(0.1)
        print(f"dropped: status={cap.stats.status.value} after {time.time() - t0:.1f}s error={cap.stats.error!r}")
        self.assertEqual(cap.stats.status, Status.RECONNECTING)

        frames_before = cap.stats.frames
        srv = serve(port)
        t0 = time.time()
        while (cap.stats.status != Status.LIVE or cap.stats.frames < frames_before + 20) and time.time() - t0 < 30:
            time.sleep(0.1)
        print(f"restored: status={cap.stats.status.value} reconnects={cap.stats.reconnects} "
              f"after {time.time() - t0:.1f}s frames={cap.stats.frames}")
        self.assertEqual(cap.stats.status, Status.LIVE)
        self.assertGreaterEqual(cap.stats.reconnects, 1)
        cap.stop()
        srv.stopping = True
        srv.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
