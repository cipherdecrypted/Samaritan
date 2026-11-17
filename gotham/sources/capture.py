from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .. import config


class Kind(str, Enum):
    CAMERA = "camera"
    STREAM = "stream"
    FILE = "file"


class Status(str, Enum):
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    PAUSED = "paused"
    ENDED = "ended"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class SourceSpec:
    kind: Kind
    ref: str
    name: str = ""
    width: int = 1280
    height: int = 720
    loop: bool = True
    camera: object = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.kind == Kind.FILE:
            return Path(self.ref).name
        return self.ref


@dataclass
class CaptureStats:
    status: Status = Status.CONNECTING
    fps: float = 0.0
    frames: int = 0
    dropped: int = 0
    reconnects: int = 0
    width: int = 0
    height: int = 0
    last_frame_age: float = 0.0
    error: str = ""
    backend: str = ""
    connected_at: float | None = None
    position: float = 0.0


@dataclass
class Frame:
    image: np.ndarray
    t: float
    seq: int


class Capture:
    def __init__(self, spec: SourceSpec):
        self.spec = spec
        self.stats = CaptureStats()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._seek: float | None = None
        self._times: deque = deque(maxlen=60)
        self._thread = threading.Thread(target=self._run, name=f"capture:{spec.label}", daemon=True)
        self._consumed_seq = -1

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3)
        self.stats.status = Status.STOPPED

    def set_paused(self, paused: bool):
        (self._paused.set if paused else self._paused.clear)()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def seek(self, fraction: float):
        self._seek = float(np.clip(fraction, 0, 1))

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def wait_next(self, last_seq: int, timeout: float = 0.5) -> Frame | None:
        with self._cond:
            if self._frame is None or self._frame.seq == last_seq:
                self._cond.wait(timeout)
            f = self._frame
            if f is not None and f.seq != last_seq:
                if last_seq >= 0 and f.seq > last_seq + 1:
                    self.stats.dropped += f.seq - last_seq - 1
                return f
            return None

    def _open(self) -> cv2.VideoCapture:
        s = self.spec
        if s.kind == Kind.CAMERA:
            from .discovery import open_camera, resolve_camera
            if s.camera is None:
                s.camera = resolve_camera(int(s.ref))
            info = s.camera
            cap = open_camera(info, s.width, s.height)
        elif s.kind == Kind.STREAM:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000]
            cap = cv2.VideoCapture(s.ref, cv2.CAP_FFMPEG, params)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(s.ref)
        return cap

    def _release(self, cap):
        if self.spec.kind == Kind.CAMERA:
            from .discovery import release
            release(cap)
        else:
            cap.release()

    def _publish(self, img: np.ndarray, t: float):
        with self._cond:
            seq = (self._frame.seq + 1) if self._frame else 0
            self._frame = Frame(img, t, seq)
            self._cond.notify_all()
        st = self.stats
        st.frames += 1
        st.height, st.width = img.shape[:2]
        self._times.append(t)
        if len(self._times) > 1:
            st.fps = (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)

    def _run(self):
        s, st = self.spec, self.stats
        backoff = 1.0
        while not self._stop.is_set():
            st.status = Status.CONNECTING if st.reconnects == 0 and st.frames == 0 else Status.RECONNECTING
            cap = self._open()
            if not cap.isOpened():
                self._release(cap)
                st.error = "unable to open source"
                if s.kind == Kind.FILE:
                    st.status = Status.FAILED
                    return
                st.reconnects += 1
                st.status = Status.RECONNECTING
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            st.backend = cap.getBackendName()
            st.error = ""
            st.connected_at = time.monotonic()
            st.status = Status.LIVE
            backoff = 1.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not (1.0 <= fps <= 240.0):
                fps = 30.0
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if s.kind == Kind.FILE else 0
            period = 1.0 / fps
            next_t = time.monotonic()
            fails = 0
            while not self._stop.is_set():
                if self._paused.is_set():
                    st.status = Status.PAUSED
                    self._stop.wait(0.05)
                    next_t = time.monotonic()
                    continue
                if st.status == Status.PAUSED:
                    st.status = Status.LIVE
                if s.kind == Kind.FILE and self._seek is not None and n_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._seek * (n_frames - 1)))
                    self._seek = None
                ok, img = cap.read()
                now = time.monotonic()
                if not ok or img is None:
                    if s.kind == Kind.FILE:
                        if s.loop and st.frames > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        st.status = Status.ENDED
                        self._release(cap)
                        return
                    fails += 1
                    if fails >= 15 or s.kind == Kind.STREAM:
                        st.error = "signal lost"
                        break
                    self._stop.wait(0.02)
                    continue
                fails = 0
                if s.kind == Kind.FILE and n_frames > 0:
                    st.position = cap.get(cv2.CAP_PROP_POS_FRAMES) / n_frames
                self._publish(img, now)
                if s.kind == Kind.FILE:
                    next_t += period
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                    elif delay < -0.25:
                        next_t = time.monotonic()
            self._release(cap)
            if not self._stop.is_set():
                st.reconnects += 1
                st.status = Status.RECONNECTING
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
