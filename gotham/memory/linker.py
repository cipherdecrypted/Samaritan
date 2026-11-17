from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..analysis.geometry import canonical_mesh
from ..inference.models import YUNET, DetectorProfile
from .store import MemoryStore


def wall(t_mono: float) -> float:
    return time.time() - (time.monotonic() - t_mono)


def face_thumb(frame: np.ndarray, box, out: int = 128) -> np.ndarray:
    x, y, w, h = box
    side = max(w, h) * 1.45
    s = out / side
    M = np.array([[s, 0, out / 2 - s * (x + w / 2)], [0, s, out / 2 - s * (y + h / 2)]], np.float32)
    return cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT)


@dataclass
class _TrackMem:
    samples: list = field(default_factory=list)
    person: int | None = None
    score: float = 0.0
    state: str = "matching"
    start: float = 0.0
    end: float = 0.0
    frames: int = 0
    thumb: np.ndarray | None = None
    thumb_q: float = 0.0
    learned: int = 0
    closed: bool = False
    sid: int | None = None
    flushed: float = 0.0
    det_scores: list = field(default_factory=list)
    meshed: int = 0
    recent: list = field(default_factory=list)
    enrolled: float = 0.0
    visible: bool = True
    mesh: np.ndarray | None = None
    mesh_q: float = 0.0
    mesh_saved: float = 0.0
    cand: dict = field(default_factory=dict)
    frames_base: int = 0
    score_base: float = 0.0


class MemoryLinker:
    MIN_KNOWN = 2
    NEW_AFTER = 3
    MID_NEW_AFTER = 6
    FORCE_NEW_AFTER = 12
    FLUSH_EVERY = 2.0
    NEW_LABEL_SECONDS = 5.0
    POSSIBLE_BELOW = 0.07

    def __init__(self, store: MemoryStore, source: str, emit=None, detector: DetectorProfile | None = None):
        self.store = store
        self.source = source
        self.emit = emit or (lambda kind, track, text, person: None)
        self.det = detector or YUNET
        self.frame: np.ndarray | None = None
        self._mem: dict[int, _TrackMem] = {}

    @property
    def th(self):
        return self.store.th

    def on_sample(self, track, emb: np.ndarray):
        if not self.store.enabled or track.id == 0:
            return
        m = self._mem.setdefault(track.uid, _TrackMem(start=track.first_seen))
        f = track.face
        q = float(f.quality.get("overall", 0))
        yaw = float(f.pose.yaw) if f.pose else 0.0
        rq = float(getattr(f, "recog_q", 1.0))
        if m.person is None:
            self._keep_best_face(m, f, q, rq, yaw)
        good = (f.score >= self.det.memory_min and q >= 0.15 and f.box[3] >= 28 and abs(yaw) <= 60 and rq >= 0.2)
        if not good:
            return
        view = face_thumb(self.frame, f.box, 72) if self.frame is not None else None
        m.samples.append((emb, yaw, q, view, rq * (0.5 + 0.5 * q)))
        m.samples = m.samples[-24:]
        m.det_scores.append(float(f.score))
        m.meshed += f.mesh is not None
        if m.state in ("matching", "uncertain"):
            self._decide(track, m)
        elif m.person is not None:
            self._verify_and_learn(track, m, emb, yaw, q, view)
        track.person_id, track.person_score, track.memory_state = m.person, m.score, m.state

    @staticmethod
    def _visit_score(m: _TrackMem) -> float:
        own = m.score if m.state == "known" else (float(np.mean(m.recent)) if m.recent else 0.0)
        return max(m.score_base, own)

    def _keep_best_face(self, m: _TrackMem, f, q: float, rq: float, yaw: float):
        if self.frame is not None and q * rq > m.thumb_q and abs(yaw) < 30:
            m.thumb, m.thumb_q = face_thumb(self.frame, f.box), q * rq
        pitch = float(f.pose.pitch) if f.pose else 90.0
        if f.mesh is not None and abs(yaw) < 15 and abs(pitch) < 15 and f.box[3] >= 60 and q * rq > m.mesh_q:
            m.mesh, m.mesh_q = canonical_mesh(f.mesh), q * rq

    def _verify_and_learn(self, track, m: _TrackMem, emb, yaw, q, view):
        th = self.th
        s = self.store.sample_score(m.person, emb)
        if s >= th.match - 0.05:
            self._keep_best_face(m, track.face, q, float(getattr(track.face, "recog_q", 1.0)), yaw)
        if s >= th.learn and m.learned < 40:
            if self.store.learn(m.person, emb, yaw, q, view):
                m.learned += 1
        m.score = 0.8 * m.score + 0.2 * s
        m.recent = (m.recent + [s])[-6:]
        if len(m.recent) >= 4 and float(np.mean(m.recent)) < th.drop:
            old = m.person
            self.emit("unlinked", track, f"{track.label} no longer matches P-{old:04d}; re-identifying", old)
            self._close_sighting(m, track)
            m.person, m.state, m.score, m.sid = None, "matching", 0.0, None
            m.samples, m.recent, m.det_scores = m.samples[-2:], [], m.det_scores[-2:]
            m.meshed = 0
            m.start = track.last_seen
            m.thumb, m.thumb_q, m.mesh, m.mesh_q, m.mesh_saved, m.frames = None, 0.0, None, 0.0, 0.0, 0
            m.frames_base, m.score_base = 0, 0.0

    def _enrol_ok(self, m: _TrackMem) -> bool:
        det = float(np.mean(m.det_scores)) if m.det_scores else 0.0
        return det >= self.det.enrol_min and (m.meshed > 0 or det >= self.det.enrol_min + 0.08)

    def _decide(self, track, m: _TrackMem):
        th = self.th
        n = len(m.samples)
        if n < self.MIN_KNOWN:
            return
        embs = np.stack([s[0] for s in m.samples])
        weights = np.array([s[4] for s in m.samples])
        busy = {mm.person for uid, mm in self._mem.items()
                if uid != track.uid and mm.person and not mm.closed and mm.visible
                and (not mm.recent or mm.recent[-1] >= th.match - 0.1)}
        ranked = self.store.match(embs, exclude=busy, weights=weights)
        best_pid, best = ranked[0] if ranked else (None, -1.0)
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        split = (len(ranked) > 1 and best >= th.match and best - second < th.margin
                 and self.store.profile_similarity(best_pid, ranked[1][0]) >= th.match)
        if best >= th.match and (best - second >= th.margin or split):
            m.person, m.score, m.state, m.recent = best_pid, best, "known", []
            self._drop_candidate(m, best_pid)
            self._save_mesh(m)
            p = self.store.person(best_pid)
            self.emit("recognized", track, f"{track.label} recognised as {p.name if p else best_pid} ({best:.2f})",
                      best_pid)
            for emb, yaw, q, view, _ in m.samples:
                if self.store.sample_score(best_pid, emb) >= th.learn:
                    self.store.learn(best_pid, emb, yaw, q, view)
            return
        stranger = (best < th.new and n >= self.NEW_AFTER) or \
                   (best < th.match - 0.06 and n >= self.MID_NEW_AFTER) or \
                   (best < th.match and n >= self.FORCE_NEW_AFTER)
        if stranger and self._enrol_ok(m):
            ordered = sorted(self._consistent(m.samples), key=lambda s: -s[4])
            pid = self.store.create_person([s[:4] for s in ordered[:6]], m.thumb, m.thumb_q, wall(m.start))
            m.person, m.score, m.state, m.enrolled, m.recent = pid, 1.0, "new", time.monotonic(), []
            self.emit("new_person", track, f"{track.label} enrolled as P-{pid:04d}", pid)
            if best_pid is not None and best >= th.match - self.POSSIBLE_BELOW:
                self._upsert_candidate(track, m, best_pid, best)
            for e in m.cand.values():
                self.store.update_candidate(e[0], other=pid)
            self._save_mesh(m)
        else:
            m.state, m.score = "uncertain", max(best, 0.0)
            if best_pid is not None and best >= th.match - self.POSSIBLE_BELOW:
                self._upsert_candidate(track, m, best_pid, best)

    def _consistent(self, samples: list) -> list:
        if len(samples) < 3:
            return samples
        E = np.stack([s[0] for s in samples])
        g = E @ E.T
        centre = int(np.argmax(g.mean(axis=1)))
        keep = [s for s, sim in zip(samples, g[centre]) if sim >= self.th.new]
        return keep if len(keep) >= 2 else samples

    def _upsert_candidate(self, track, m: _TrackMem, pid: int, score: float):
        e = m.cand.get(pid)
        if e is None:
            thumb = m.thumb if m.thumb is not None else (face_thumb(self.frame, track.face.box)
                                                         if self.frame is not None else None)
            cid = self.store.add_candidate(pid, self.source, wall(track.last_seen), score, track.label, thumb)
            m.cand[pid] = [cid, score]
            p = self.store.person(pid)
            self.emit("possible_match", track, f"{track.label} possibly {p.name if p else pid} ({score:.2f})", pid)
        elif score > e[1] + 0.01:
            self.store.update_candidate(e[0], score=score, thumb=m.thumb)
            e[1] = score

    def _drop_candidate(self, m: _TrackMem, pid: int):
        e = m.cand.pop(pid, None)
        if e is not None:
            self.store.delete_candidate(e[0])

    def _save_mesh(self, m: _TrackMem):
        if m.person is not None and m.mesh is not None and m.mesh_q > m.mesh_saved + 0.02:
            self.store.update_mesh(m.person, m.mesh, m.mesh_q)
            m.mesh_saved = m.mesh_q

    def on_frame(self, tracks):
        now = time.monotonic()
        for tr in tracks:
            m = self._mem.get(tr.uid)
            if m is None:
                continue
            m.visible = bool(tr.visible)
            if tr.visible:
                m.end = tr.last_seen
                m.frames += 1
            if m.state == "new" and now - m.enrolled > self.NEW_LABEL_SECONDS:
                m.state = "known"
            tr.person_id, tr.person_score, tr.memory_state = m.person, m.score, m.state
            if m.person is None or m.closed:
                continue
            if m.sid is None:
                prev = self.store.resume_visit(m.person, self.source, wall(m.start))
                if prev is not None:
                    m.sid, m.frames_base, m.score_base = prev["id"], prev["frames"], prev["score"]
                    self.store.update_sighting(m.sid, wall(m.end or now), m.frames_base + m.frames, self._visit_score(m))
                else:
                    m.sid = self.store.add_sighting(m.person, "live", self.source, wall(m.start), wall(m.end or now),
                                                    m.frames, self._visit_score(m), m.thumb)
                m.flushed = now
            elif now - m.flushed >= self.FLUSH_EVERY and tr.visible:
                self.store.update_sighting(m.sid, wall(m.end), m.frames_base + m.frames, self._visit_score(m))
                self._save_mesh(m)
                m.flushed = now

    def _close_sighting(self, m: _TrackMem, track):
        if m.person is None:
            return
        end = m.end or track.last_seen
        if m.sid is None:
            self.store.add_sighting(m.person, "live", self.source, wall(m.start), wall(end), m.frames,
                                    self._visit_score(m), m.thumb)
        else:
            self.store.update_sighting(m.sid, wall(end), m.frames_base + m.frames, self._visit_score(m), m.thumb)
        if m.thumb is not None:
            self.store.update_thumb(m.person, m.thumb, m.thumb_q)
        self._save_mesh(m)

    def on_end(self, track):
        m = self._mem.pop(track.uid, None)
        if m is None or m.closed or m.person is None:
            return
        m.closed = True
        self._close_sighting(m, track)

    def close_all(self, tracks):
        for tr in list(tracks):
            self.on_end(tr)
