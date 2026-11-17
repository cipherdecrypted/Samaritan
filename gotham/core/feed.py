from __future__ import annotations

import itertools
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..analysis.pipeline import AnalysisSettings, Face, FaceAnalyzer
from ..inference.models import make_embedder
from ..sources.capture import Capture, CaptureStats, Kind, SourceSpec, Status
from ..tracking.tracker import FaceTracker, State, Track

_feed_ids = itertools.count(1)


@dataclass(frozen=True)
class TrackView:
    id: int
    label: str
    state: str
    box: np.ndarray
    face: Face
    visible: bool
    trail: tuple
    velocity: np.ndarray
    first_seen: float
    last_seen: float
    frames_seen: int
    reacquired: int
    suppressed: bool
    person_id: int | None = None
    person_score: float = 0.0
    memory_state: str = ""

    @staticmethod
    def of(t: Track) -> "TrackView":
        return TrackView(t.id, t.label, t.state.value, t.box.copy(), t.face, t.visible, tuple(t.trail),
                         t.velocity.copy(), t.first_seen, t.last_seen, t.frames_seen, t.reacquired, t.suppressed,
                         t.person_id, t.person_score, t.memory_state)


@dataclass
class FeedResult:
    seq: int
    frame: np.ndarray
    t: float
    faces: list[Face]
    tracks: list[TrackView]
    timings: dict[str, float]
    process_fps: float
    latency_ms: float


@dataclass
class FeedEvent:
    t: float
    wall: float
    feed_id: int
    kind: str
    subject: int | None = None
    text: str = ""
    person: int | None = None


@dataclass
class Feed:
    spec: SourceSpec
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    id: int = field(default_factory=lambda: next(_feed_ids))

    def __post_init__(self):
        self.capture = Capture(self.spec)
        self.settings.reuse_landmarks = True
        self.tracker = FaceTracker()
        self.result: FeedResult | None = None
        self.analysis_enabled = True
        self.events: deque[FeedEvent] = deque(maxlen=2000)
        self.on_event = None
        self.memory = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tracking_cmd: str | None = None
        self._remove_cmd: list[int] = []
        self._suppress_cmd: list[tuple[int, bool]] = []
        self._thread = threading.Thread(target=self._run, name=f"feed:{self.spec.label}", daemon=True)
        self._proc_times: deque = deque(maxlen=30)
        self._prev_states: dict[int, tuple[str, int]] = {}
        self.faults = 0
        self._consecutive_faults = 0
        self._seq = 0
        self.provider = ""

    @property
    def name(self) -> str:
        return f"FEED-{self.id:02d}"

    @property
    def stats(self) -> CaptureStats:
        return self.capture.stats

    def start(self):
        self.capture.start()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.capture.stop()
        self._thread.join(timeout=3)
        self.tracker.stop()

    def start_tracking(self):
        self._tracking_cmd = "start"

    def stop_tracking(self):
        self._tracking_cmd = "stop"

    @property
    def tracking(self) -> bool:
        return self.tracker.active

    def drop_subject(self, sid: int):
        self._remove_cmd.append(sid)

    def suppress_subject(self, sid: int, value: bool = True):
        self._suppress_cmd.append((sid, value))

    def latest(self) -> FeedResult | None:
        with self._lock:
            return self.result

    def _emit(self, kind: str, subject: int | None = None, text: str = "", person: int | None = None):
        if person is None and subject:
            tr = self.tracker.get(subject)
            person = tr.person_id if tr else None
        ev = FeedEvent(time.monotonic(), time.time(), self.id, kind, subject, text, person)
        self.events.append(ev)
        cb = self.on_event
        if cb:
            try:
                cb(self, ev)
            except Exception:
                pass

    def _diff_tracks(self, tracks: list[Track]):
        now_ids = {}
        for t in tracks:
            if not t.id:
                continue
            now_ids[t.id] = (t.state.value, t.reacquired)
            prev = self._prev_states.get(t.id)
            if prev is None:
                if t.reacquired == 0:
                    self._emit("acquired", t.id, f"{t.label} acquired")
                continue
            pstate, preacq = prev
            if t.reacquired > preacq:
                self._emit("reacquired", t.id, f"{t.label} reacquired")
            elif pstate != t.state.value:
                if t.state == State.COASTING:
                    self._emit("occluded", t.id, f"{t.label} occluded")
                elif t.state == State.LOST:
                    self._emit("lost", t.id, f"{t.label} left view")
                elif t.state == State.ACTIVE and pstate == State.COASTING.value:
                    self._emit("resumed", t.id, f"{t.label} visible")
        for sid in set(self._prev_states) - set(now_ids):
            self._emit("expired", sid, f"S{sid:02d} released")
        self._prev_states = now_ids

    def _run(self):
        analyzer = FaceAnalyzer()
        embedder = make_embedder()
        self.tracker.cfg.reacquire_similarity = embedder.profile.reacquire
        self.recogniser = embedder.profile.display
        self.provider = analyzer.provider
        linker = None
        if self.memory is not None:
            from ..memory.linker import MemoryLinker
            linker = MemoryLinker(self.memory, self.spec.label,
                                  emit=lambda kind, tr, text, person: self._emit(kind, tr.id, text, person),
                                  detector=analyzer.detector.profile)
            self.tracker.on_sample = linker.on_sample
            self.tracker.on_end = linker.on_end
        last_seq = -1
        last_status = None
        seq = 0
        try:
            while not self._stop.is_set():
                st = self.capture.stats.status
                if st != last_status:
                    if st == Status.LIVE and last_status in (None, Status.CONNECTING, Status.RECONNECTING):
                        self._emit("connected", None, "signal acquired")
                    elif st == Status.RECONNECTING and last_status == Status.LIVE:
                        self._emit("signal_lost", None, "signal lost")
                    elif st == Status.FAILED:
                        self._emit("signal_lost", None, "source unavailable")
                    last_status = st
                self._apply_commands()
                fr = self.capture.wait_next(last_seq, timeout=0.25)
                if fr is None:
                    continue
                last_seq = fr.seq
                try:
                    self._process(fr, analyzer, embedder, linker)
                    self._consecutive_faults = 0
                except Exception as e:
                    self.faults += 1
                    self._consecutive_faults += 1
                    if self.faults <= 5 or self.faults % 100 == 0:
                        import traceback
                        print(f"[{self.name}] frame fault #{self.faults}: {e!r}", flush=True)
                        traceback.print_exc()
                    if self.faults == 1:
                        self._emit("fault", None, f"analysis fault (recovering): {type(e).__name__}")
                    if self._consecutive_faults >= 25:
                        self._consecutive_faults = 0
                        try:
                            analyzer.close()
                        except Exception:
                            pass
                        analyzer = FaceAnalyzer()
                        embedder = make_embedder()
                        self._emit("fault", None, "analysis models reloaded")
        finally:
            analyzer.close()

    def _process(self, fr, analyzer, embedder, linker):
        t0 = time.perf_counter()
        if self.analysis_enabled:
            res = analyzer.analyze(fr.image, self.settings)
            faces, timings = res.faces, res.timings
        else:
            faces, timings = [], {}
        tracks: list[Track] = []
        if self.tracker.active:
            img = fr.image
            if linker is not None:
                linker.frame = img
            self.tracker.memory_active = linker is not None and self.memory.enabled

            def _embed(f, im=img):
                v, q = embedder.embed_q(im, f.kps)
                f.recog_q = q
                return v

            def _embed_many(fs, im=img):
                out = []
                for f, (v, q) in zip(fs, embedder.embed_q_many(im, [f.kps for f in fs])):
                    f.recog_q = q
                    out.append(v)
                return out
            tracks = self.tracker.update(faces, fr.t, embed=_embed,
                                         embed_many=None if os.environ.get("GOTHAM_NO_PREFETCH") else _embed_many)
            if linker is not None:
                linker.on_frame(tracks)
            self._diff_tracks(tracks)
        views = [TrackView.of(t) for t in tracks if t.id]
        now = time.perf_counter()
        self._proc_times.append(now)
        pf = 0.0
        if len(self._proc_times) > 1:
            pf = (len(self._proc_times) - 1) / max(self._proc_times[-1] - self._proc_times[0], 1e-6)
        timings["total"] = (now - t0) * 1e3
        self._seq += 1
        result = FeedResult(self._seq, fr.image, fr.t, faces, views, timings, pf,
                            (time.monotonic() - fr.t) * 1e3)
        with self._lock:
            self.result = result

    def _apply_commands(self):
        cmd, self._tracking_cmd = self._tracking_cmd, None
        if cmd == "start":
            self.tracker.start(time.monotonic())
            self._prev_states = {}
            self._emit("session_start", None, "tracking session started")
        elif cmd == "stop":
            self.tracker.stop()
            self._prev_states = {}
            self._emit("session_stop", None, "tracking session ended")
        while self._remove_cmd:
            sid = self._remove_cmd.pop()
            self.tracker.remove(sid)
        while self._suppress_cmd:
            sid, v = self._suppress_cmd.pop()
            self.tracker.set_suppressed(sid, v)

    @property
    def is_file(self) -> bool:
        return self.spec.kind == Kind.FILE
