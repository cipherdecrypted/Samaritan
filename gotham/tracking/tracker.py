from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from ..analysis.pipeline import Face
from .assignment import linear_sum_assignment


class State(str, Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    COASTING = "coasting"
    LOST = "lost"


class Kalman:
    def __init__(self, cx: float, cy: float, s: float):
        self.s_ref = float(s)
        self.x = np.array([cx, cy, s, 0, 0, 0], np.float64)
        self.P = np.diag([s * 0.1, s * 0.1, s * 0.1, s * 2, s * 2, s * 0.5]) ** 2

    def predict(self, dt: float, coasting: bool = False):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        s = max(self.x[2], 8.0)
        q_pos, q_vel = (0.05 * s) ** 2 * dt, (1.5 * s) ** 2 * dt
        Q = np.diag([q_pos, q_pos, q_pos * 0.5, q_vel, q_vel, q_vel * 0.25])
        if coasting:
            self.x[3:5] *= 0.85 ** (dt * 30)
            self.x[5] = 0.0
        self.x = F @ self.x
        self.x[2] = float(np.clip(self.x[2], max(4.0, 0.7 * self.s_ref), 1.4 * self.s_ref))
        vmax = 4.0 * self.s_ref
        sp = float(np.hypot(self.x[3], self.x[4]))
        if sp > vmax:
            self.x[3:5] *= vmax / sp
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray):
        self.s_ref = float(z[2])
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1
        s = max(self.x[2], 8.0)
        R = np.diag([0.04 * s, 0.04 * s, 0.06 * s]) ** 2
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


@dataclass
class Track:
    uid: int
    kf: Kalman
    aspect: float
    face: Face
    first_seen: float
    last_seen: float
    state: State = State.TENTATIVE
    hits: int = 1
    misses: int = 0
    frames_seen: int = 1
    reacquired: int = 0
    suppressed: bool = False
    trail: deque = field(default_factory=lambda: deque(maxlen=240))
    gallery: list = field(default_factory=list)
    _since_embed: int = 0
    id: int = 0
    person_id: int | None = None
    person_score: float = 0.0
    memory_state: str = ""

    @property
    def label(self) -> str:
        return f"S{self.id:02d}"

    @property
    def box(self) -> np.ndarray:
        if self.state in (State.ACTIVE, State.TENTATIVE) and self.misses == 0:
            return self.face.box
        return self.predicted_box

    @property
    def predicted_box(self) -> np.ndarray:
        cx, cy, s = self.kf.x[:3]
        w = s * self.aspect
        return np.array([cx - w / 2, cy - s / 2, w, s], np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[3:5].astype(np.float32)

    @property
    def visible(self) -> bool:
        return self.state in (State.ACTIVE, State.TENTATIVE) and self.misses <= self.grace

    grace: int = 0

    def similarity(self, emb: np.ndarray) -> float:
        if not self.gallery:
            return -1.0
        return float(max(g @ emb for g in self.gallery))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackerConfig:
    confirm_hits: int = 3
    coast_seconds: float = 1.5
    lost_ttl: float = 120.0
    reacquire_similarity: float = 0.50
    reacquire_min_size: float = 32.0
    gate: float = 1.2
    trail_seconds: float = 6.0
    gallery_size: int = 6
    grace_misses: int = 4
    appearance_weight: float = 0.35
    tiebreak_budget: int = 3
    tiebreak_max_faces: int = 8
    embed_every: int = 10
    embed_fast_every: int = 2
    embed_budget: int = 3
    reacquire_budget: int = 4


class FaceTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self.active = False
        self.started_at: float | None = None
        self._ids = itertools.count(1)
        self._uids = itertools.count(1)
        self._last_t: float | None = None
        self.total_subjects = 0
        self.total_reacquisitions = 0
        self.on_sample: Callable[[Track, np.ndarray], None] | None = None
        self.on_end: Callable[[Track], None] | None = None
        self.memory_active = False

    def _ended(self, tr: Track):
        if self.on_end is not None:
            try:
                self.on_end(tr)
            except Exception:
                pass

    def start(self, t: float):
        self.reset()
        self.active = True
        self.started_at = t

    def stop(self):
        self.reset()
        self.active = False

    def reset(self):
        for tr in self.tracks:
            self._ended(tr)
            tr.gallery.clear()
        self.tracks = []
        self._ids = itertools.count(1)
        self._uids = itertools.count(1)
        self._last_t = None
        self.started_at = None
        self.total_subjects = 0
        self.total_reacquisitions = 0

    def remove(self, track_id: int):
        for t in self.tracks:
            if t.id == track_id:
                self._ended(t)
        self.tracks = [t for t in self.tracks if t.id != track_id]

    def set_suppressed(self, track_id: int, value: bool):
        for t in self.tracks:
            if t.id == track_id:
                t.suppressed = value

    def get(self, track_id: int) -> Track | None:
        return next((t for t in self.tracks if t.id == track_id and track_id), None)

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.id]

    def update(self, faces: list[Face], t: float,
               embed: Callable[[Face], np.ndarray | None] | None = None,
               embed_many: Callable[[list[Face]], list] | None = None) -> list[Track]:
        if not self.active:
            return []
        dt = 1 / 30 if self._last_t is None else float(np.clip(t - self._last_t, 1e-3, 1.0))
        self._last_t = t
        cfg = self.cfg
        for tr in self.tracks:
            if tr.state != State.LOST:
                tr.kf.predict(dt, coasting=tr.misses > 0)

        live = [tr for tr in self.tracks if tr.state != State.LOST]
        if embed_many is not None and faces:
            self._prefetch(live, faces, embed_many)
        matches, um_t, um_d = self._associate(live, faces, embed)

        for ti, di in matches:
            self._apply(live[ti], faces[di], t)

        for ti in um_t:
            tr = live[ti]
            tr.misses += 1
            if tr.state == State.TENTATIVE:
                tr.state = State.LOST
                tr.gallery.clear()
            elif tr.state == State.ACTIVE and tr.misses <= cfg.grace_misses:
                pass
            elif t - tr.last_seen <= cfg.coast_seconds:
                tr.state = State.COASTING
            else:
                tr.state = State.LOST

        candidates = [tr for tr in self.tracks if tr.state in (State.LOST, State.COASTING) and tr.gallery]
        new_faces = [faces[i] for i in um_d]
        if candidates and new_faces and embed is not None:
            order = sorted(range(len(new_faces)), key=lambda j: -new_faces[j].box[3])[:cfg.reacquire_budget]
            embs = [self._emb(f, embed) if (j in order and f.box[3] >= cfg.reacquire_min_size) else None
                    for j, f in enumerate(new_faces)]
            sims = np.full((len(candidates), len(new_faces)), -1.0)
            for i, tr in enumerate(candidates):
                for j, e in enumerate(embs):
                    if e is not None:
                        sims[i, j] = tr.similarity(e)
            used = set()
            for i, j in linear_sum_assignment(1.0 - sims):
                if sims[i, j] >= cfg.reacquire_similarity:
                    tr, f = candidates[i], new_faces[j]
                    self._reinit(tr, f, t)
                    if embs[j] is not None:
                        self._add_embedding(tr, embs[j])
                    used.add(j)
            new_faces = [f for j, f in enumerate(new_faces) if j not in used]

        for f in new_faces:
            self._spawn(f, t)

        if embed is not None:
            due = []
            for tr in self.tracks:
                if tr.state == State.ACTIVE and tr.misses == 0:
                    tr._since_embed += 1
                    fast = self.memory_active and tr.memory_state in ("", "matching", "uncertain")
                    every = cfg.embed_fast_every if fast else cfg.embed_every
                    q = tr.face.quality
                    if (not tr.gallery or tr._since_embed >= every) and q.get("frontality", 0) > 0.2 \
                            and tr.face.box[3] >= 26:
                        due.append((0 if fast else 1, -tr._since_embed, tr.uid, tr))
            due.sort(key=lambda d: d[:3])
            for *_, tr in due[:cfg.embed_budget]:
                e = self._emb(tr.face, embed)
                if e is not None:
                    self._add_embedding(tr, e)
                tr._since_embed = 0

        keep = []
        for tr in self.tracks:
            if tr.state == State.LOST and tr.id == 0:
                continue
            if tr.state == State.LOST and (t - tr.last_seen > cfg.lost_ttl or not tr.gallery):
                self._ended(tr)
                tr.gallery.clear()
                continue
            keep.append(tr)
        self.tracks = keep

        for tr in self.tracks:
            while tr.trail and t - tr.trail[0][0] > cfg.trail_seconds:
                tr.trail.popleft()
        return self.tracks

    @staticmethod
    def _emb(f: Face, embed):
        e = getattr(f, "_emb", None)
        if e is None:
            e = embed(f)
            f._emb = e
        return e

    def _prefetch(self, live: list[Track], faces: list[Face], embed_many):
        cfg = self.cfg
        C, BIG = self._position_costs(live, faces) if live else (np.zeros((0, len(faces))), 1e6)
        pairs = [(i, j) for i, j in linear_sum_assignment(C) if C[i, j] < BIG] if live else []
        matched = {j for _, j in pairs}
        due = []
        for i, j in pairs:
            tr, f = live[i], faces[j]
            if tr.state != State.ACTIVE or f.quality.get("frontality", 0) <= 0.2 or f.box[3] < 26:
                continue
            fast = self.memory_active and tr.memory_state in ("", "matching", "uncertain")
            every = cfg.embed_fast_every if fast else cfg.embed_every
            if not tr.gallery or tr._since_embed + 1 >= every:
                due.append((0 if fast else 1, -(tr._since_embed + 1), tr.uid, j))
        due.sort(key=lambda d: d[:3])
        want = [j for *_, j in due[:cfg.embed_budget]]
        if any(tr.state in (State.LOST, State.COASTING) and tr.gallery for tr in self.tracks):
            um = [j for j in range(len(faces)) if j not in matched and faces[j].box[3] >= cfg.reacquire_min_size]
            want += sorted(um, key=lambda j: -faces[j].box[3])[:cfg.reacquire_budget]
        if live and len(faces) <= cfg.tiebreak_max_faces:
            want += self._tiebreak_faces(live, faces, C, BIG)
        todo = [j for j in dict.fromkeys(want) if getattr(faces[j], "_emb", None) is None]
        if todo:
            for j, e in zip(todo, embed_many([faces[j] for j in todo])):
                faces[j]._emb = e

    def _position_costs(self, tracks: list[Track], faces: list[Face]):
        BIG = 1e6
        C = np.full((len(tracks), len(faces)), BIG)
        for i, tr in enumerate(tracks):
            pb = tr.predicted_box
            pc = pb[:2] + pb[2:] / 2
            size = max(pb[3], 8.0)
            gate = self.cfg.gate * (1.0 + 0.25 * min(tr.misses ** 0.5, 2.0))
            for j, f in enumerate(faces):
                d = float(np.linalg.norm(f.centre - pc)) / size
                if d > gate:
                    continue
                ratio = f.box[3] / size
                if ratio < 0.5 or ratio > 2.0:
                    continue
                C[i, j] = 0.5 * (1 - _iou(pb, f.box)) + 0.5 * min(1.0, d / gate)
        return C, BIG

    def _associate(self, tracks: list[Track], faces: list[Face], embed=None):
        if not tracks or not faces:
            return [], list(range(len(tracks))), list(range(len(faces)))
        C, BIG = self._position_costs(tracks, faces)
        if embed is not None and len(faces) <= self.cfg.tiebreak_max_faces:
            self._appearance_tiebreak(tracks, faces, C, BIG, embed)
        pairs = [(i, j) for i, j in linear_sum_assignment(C) if C[i, j] < BIG]
        mt = {i for i, _ in pairs}
        md = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(faces)) if j not in md]

    def _tiebreak_faces(self, tracks, faces, C, BIG) -> list[int]:
        ok = C < BIG
        rows = [i for i in range(len(tracks)) if ok[i].sum() >= 2 and tracks[i].gallery]
        cols = [j for j in range(len(faces)) if ok[:, j].sum() >= 2]
        js = sorted({j for i in rows for j in np.flatnonzero(ok[i])} | set(cols), key=lambda j: -faces[j].box[3])
        return [j for j in js if faces[j].box[3] >= 28][:self.cfg.tiebreak_budget]

    def _appearance_tiebreak(self, tracks, faces, C, BIG, embed):
        ok = C < BIG
        js = self._tiebreak_faces(tracks, faces, C, BIG)
        if not js:
            return
        w = self.cfg.appearance_weight
        for j in js:
            e = self._emb(faces[j], embed)
            if e is None:
                continue
            for i in np.flatnonzero(ok[:, j]):
                tr = tracks[i]
                sim = tr.similarity(e) if tr.gallery else 0.35
                C[i, j] = (1 - w) * C[i, j] + w * (1.0 - max(sim, 0.0))

    def _apply(self, tr: Track, f: Face, t: float):
        x, y, w, h = f.box
        tr.kf.update(np.array([x + w / 2, y + h / 2, h]))
        tr.aspect = 0.8 * tr.aspect + 0.2 * (w / max(h, 1))
        tr.face = f
        tr.hits += 1
        tr.frames_seen += 1
        tr.misses = 0
        tr.last_seen = t
        if tr.state == State.TENTATIVE and tr.hits >= self.cfg.confirm_hits:
            tr.state = State.ACTIVE
            tr.id = next(self._ids)
            self.total_subjects += 1
        elif tr.state == State.COASTING:
            tr.state = State.ACTIVE
        f.track_id = tr.id or None
        tr.trail.append((t, float(x + w / 2), float(y + h / 2)))

    def _reinit(self, tr: Track, f: Face, t: float):
        x, y, w, h = f.box
        tr.kf = Kalman(x + w / 2, y + h / 2, h)
        tr.aspect = w / max(h, 1)
        tr.trail.clear()
        tr.reacquired += 1
        self.total_reacquisitions += 1
        tr.state = State.ACTIVE
        tr.face = f
        f.track_id = tr.id
        tr.hits += 1
        tr.frames_seen += 1
        tr.misses = 0
        tr.last_seen = t
        tr.trail.append((t, float(x + w / 2), float(y + h / 2)))

    def _spawn(self, f: Face, t: float):
        x, y, w, h = f.box
        tr = Track(uid=next(self._uids), kf=Kalman(x + w / 2, y + h / 2, h), aspect=w / max(h, 1),
                   face=f, first_seen=t, last_seen=t)
        tr.grace = self.cfg.grace_misses
        tr.trail.append((t, float(x + w / 2), float(y + h / 2)))
        self.tracks.append(tr)

    def _add_embedding(self, tr: Track, e: np.ndarray):
        tr.gallery.append(e)
        if len(tr.gallery) > self.cfg.gallery_size:
            tr.gallery.pop(1)
        if self.on_sample is not None and tr.id:
            try:
                self.on_sample(tr, e)
            except Exception:
                import traceback
                traceback.print_exc()
