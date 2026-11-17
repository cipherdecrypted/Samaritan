from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..inference.landmarks import FaceLandmarks
from ..inference.models import make_detector
from ..inference.yunet import Detection
from . import geometry as geo
from .quality import face_quality


@dataclass
class AnalysisSettings:
    threshold: float | None = None
    small_faces: bool = False
    deep_scan: bool = True
    reuse_landmarks: bool = False
    landmark_budget: int = 4
    max_reuse_age: int = 6
    landmarks: bool = True
    max_landmark_faces: int = 8
    min_landmark_size: float = 18.0
    verify_below: float | None = None


@dataclass
class Face:
    box: np.ndarray
    score: float
    kps: np.ndarray
    truncated: bool
    mesh: np.ndarray | None = None
    blend: dict[str, float] = field(default_factory=dict)
    matrix: np.ndarray | None = None
    pose: geo.Pose | None = None
    gaze: geo.Gaze | None = None
    eye_open: tuple[float, float] | None = None
    quality: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    track_id: int | None = None
    recog_q: float = 1.0

    @property
    def centre(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2], np.float32)


@dataclass
class FrameAnalysis:
    faces: list[Face]
    timings: dict[str, float]
    size: tuple[int, int]


class FaceAnalyzer:
    def __init__(self, prefer_gpu: bool = True):
        self.detector = make_detector(prefer_gpu=prefer_gpu)
        self.profile = self.detector.profile
        self._lm_workers = max(1, min(4, (os.cpu_count() or 4) // 4))
        self._pool = ThreadPoolExecutor(max_workers=self._lm_workers, thread_name_prefix="landmarks")
        self._local = threading.local()
        self._landmarkers: list[FaceLandmarks] = []
        self._lm_lock = threading.Lock()
        self._lm_cache: list[dict] = []
        self._n = 0

    @property
    def provider(self) -> str:
        return self.detector.provider

    def _fit(self, rgb: np.ndarray, box: np.ndarray):
        lm = getattr(self._local, "lm", None)
        if lm is None:
            lm = self._local.lm = FaceLandmarks()
            with self._lm_lock:
                self._landmarkers.append(lm)
        return lm.run(rgb, box)

    def close(self):
        self._pool.shutdown(wait=True)
        with self._lm_lock:
            for lm in self._landmarkers:
                lm.close()
            self._landmarkers.clear()

    def analyze(self, frame: np.ndarray, s: AnalysisSettings) -> FrameAnalysis:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        H, W = frame.shape[:2]
        t0 = time.perf_counter()
        thr = s.threshold if s.threshold is not None else self.profile.threshold
        self._n += 1
        deep = s.deep_scan and max(frame.shape[:2]) > 800 and self._n % 2 == 0
        mode = True if s.small_faces else ("auto" if deep else False)
        dets = self.detector.detect(frame, tiled=mode, threshold=thr)
        verify_below = s.verify_below if s.verify_below is not None else self.profile.verify_below
        t1 = time.perf_counter()
        faces = [self._face(d) for d in dets]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if s.landmarks and faces:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            order = sorted(range(len(faces)), key=lambda i: -faces[i].box[3])
            cache, self._lm_cache = self._lm_cache if s.reuse_landmarks else [], []
            cand = [faces[i] for i in order[: s.max_landmark_faces] if faces[i].box[3] >= s.min_landmark_size]

            def unconfirmed(f: Face) -> bool:
                return f.score < verify_below and f.box[3] >= 2 * s.min_landmark_size and not f.truncated
            if s.reuse_landmarks:
                hits = {id(f): self._match_cache(f, cache) for f in cand}
                usable = {id(f) for f in cand if hits[id(f)] is not None and hits[id(f)]["age"] < s.max_reuse_age}
                fresh = sorted((f for f in cand if id(f) not in usable), key=lambda f: not unconfirmed(f))
                stale = sorted((f for f in cand if id(f) in usable and hits[id(f)]["age"] >= 1),
                               key=lambda f: -hits[id(f)]["age"])
                todo = (fresh + stale)[:max(1, s.landmark_budget)]
                chosen = {id(f) for f in todo}
                for f in cand:
                    if id(f) in usable and id(f) not in chosen:
                        self._lm_cache.append(self._apply_cached(f, hits[id(f)]))
            else:
                todo = cand
            if len(todo) > 1:
                fits = list(self._pool.map(lambda f: self._fit(rgb, f.box), todo))
            else:
                fits = [self._fit(rgb, f.box) for f in todo]
            for f, r in zip(todo, fits):
                if r is None:
                    continue
                nose = r.points[geo.NOSE_TIP, :2]
                if np.linalg.norm(nose - f.kps[2]) > 0.45 * max(f.box[2], f.box[3]):
                    continue
                f.mesh, f.blend, f.matrix = r.points, r.blendshapes, r.matrix
                self._lm_cache.append({"box": f.box.copy(), "mesh": f.mesh, "blend": f.blend, "matrix": f.matrix,
                                       "age": 0})
            checked = {id(f) for f in cand}
            faces = [f for f in faces if not (id(f) in checked and f.mesh is None and unconfirmed(f))]
        t2 = time.perf_counter()
        for f in faces:
            self._derive(f, gray, W, H)
        t3 = time.perf_counter()
        return FrameAnalysis(faces, {"detect": (t1 - t0) * 1e3, "landmarks": (t2 - t1) * 1e3,
                                     "derive": (t3 - t2) * 1e3}, (W, H))

    @staticmethod
    def _match_cache(f: Face, cache: list[dict]) -> dict | None:
        best, best_iou = None, 0.5
        for c in cache:
            x, y, w, h = f.box
            X, Y, W2, H2 = c["box"]
            ix = max(0.0, min(x + w, X + W2) - max(x, X))
            iy = max(0.0, min(y + h, Y + H2) - max(y, Y))
            iou = ix * iy / (w * h + W2 * H2 - ix * iy + 1e-9)
            if iou > best_iou:
                best, best_iou = c, iou
        return best

    @staticmethod
    def _apply_cached(f: Face, best: dict) -> dict:
        s = f.box[3] / max(best["box"][3], 1e-6)
        old_c = best["box"][:2] + best["box"][2:] / 2
        new_c = f.box[:2] + f.box[2:] / 2
        mesh = best["mesh"].copy()
        mesh[:, :2] = (mesh[:, :2] - old_c) * s + new_c
        mesh[:, 2] *= s
        f.mesh, f.blend, f.matrix = mesh, best["blend"], best["matrix"]
        return {"box": f.box.copy(), "mesh": mesh, "blend": best["blend"], "matrix": best["matrix"],
                "age": best["age"] + 1}

    @staticmethod
    def _face(d: Detection) -> Face:
        return Face(box=d.box, score=d.score, kps=d.kps, truncated=d.truncated)

    @staticmethod
    def _derive(f: Face, gray: np.ndarray, W: int, H: int):
        if f.mesh is not None:
            f.pose = geo.pose_from_matrix(f.matrix) if f.matrix is not None else geo.pose_from_mesh(f.mesh)
            f.eye_open = geo.eye_openness(f.mesh)
            f.gaze = geo.gaze_from_mesh(f.mesh, f.pose, f.blend)
            f.metrics = geo.geometry_metrics(f.mesh)
        else:
            f.pose = geo.pose_from_5pt(f.kps, W, H)
        yaw = f.pose.yaw if f.pose else None
        pitch = f.pose.pitch if f.pose else None
        f.quality = face_quality(gray, f.box, f.score, yaw, pitch, f.mesh is not None, f.eye_open)
