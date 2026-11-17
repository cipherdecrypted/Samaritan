from __future__ import annotations

import os

import numpy as np

from . import adaface, arcface, scrfd
from .embedder import AppearanceEmbedder, align112
from .gothamfaces import GothamFaces
from .profiles import ARCFACE, FUSED, SCRFD, SFACE, YUNET, DetectorProfile, RecogniserProfile, display_name
from .runtime import run_batched, shared_session
from .yunet import YuNetDetector

__all__ = ["ARCFACE", "FUSED", "SCRFD", "SFACE", "YUNET", "DetectorProfile", "RecogniserProfile", "GothamFaces",
           "display_name", "make_detector", "make_embedder", "recogniser_profile", "run_batched"]


class _SFaceAdapter:
    profile = SFACE

    def embed_q_many(self, frame, kps_list) -> list[tuple[np.ndarray, float]]:
        return [self.embed_q(frame, k) for k in kps_list]

    def __init__(self):
        self._e = AppearanceEmbedder()
        self.session = self._e.session

    def embed_q(self, frame, kps) -> tuple[np.ndarray, float]:
        return self._e.embed(frame, kps), 1.0

    def embed(self, frame, kps) -> np.ndarray:
        return self._e.embed(frame, kps)


class _ArcFaceAdapter:
    profile = ARCFACE
    NORM_LO, NORM_HI = 16.0, 22.0
    FLIP_MAX = 4

    def __init__(self):
        self._e = arcface.ArcFaceEmbedder()
        self.session = self._e.session
        self._batch = (shared_session(arcface.ARCFACE_BATCH_MODEL)
                       if arcface.ARCFACE_BATCH_MODEL.exists() and not os.environ.get("GOTHAM_NO_BATCH") else None)

    def _run(self, crops: list[np.ndarray]) -> np.ndarray:
        if self._batch is None or len(crops) <= 3:
            return np.stack([self.session.run(self._e._blob([c]))[0][0] for c in crops]).astype(np.float32)
        return run_batched(self._batch, self._e._blob(crops))

    def embed_q_many(self, frame, kps_list) -> list[tuple[np.ndarray, float]]:
        if not kps_list:
            return []
        crops = [align112(frame, k) for k in kps_list]
        n = len(crops)
        flip = n <= self.FLIP_MAX
        raws = self._run(crops + ([c[:, ::-1].copy() for c in crops] if flip else []))
        if flip:
            v = raws[:n] + raws[n:]
            norms = (np.linalg.norm(raws[:n], axis=1) + np.linalg.norm(raws[n:], axis=1)) / 2
        else:
            v, norms = raws, np.linalg.norm(raws, axis=1)
        q = np.clip((norms - self.NORM_LO) / (self.NORM_HI - self.NORM_LO), 0.05, 1.0)
        return [(v[i] / (np.linalg.norm(v[i]) + 1e-9), float(q[i])) for i in range(n)]

    def embed_crops(self, crops: list[np.ndarray], flip: bool | None = None) -> list[tuple[np.ndarray, float]]:
        n = len(crops)
        if not n:
            return []
        flip = n <= self.FLIP_MAX if flip is None else flip
        raws = self._run(crops + ([c[:, ::-1].copy() for c in crops] if flip else []))
        v = raws[:n] + raws[n:] if flip else raws
        norms = np.linalg.norm(raws[:n], axis=1)
        q = np.clip((norms - self.NORM_LO) / (self.NORM_HI - self.NORM_LO), 0.05, 1.0)
        return [(v[i] / (np.linalg.norm(v[i]) + 1e-9), float(q[i])) for i in range(n)]

    def embed_q(self, frame, kps) -> tuple[np.ndarray, float]:
        crop = align112(frame, kps)
        raws = [self.session.run(self._e._blob([c]))[0][0] for c in (crop, crop[:, ::-1].copy())]
        n = float(np.mean([np.linalg.norm(r) for r in raws]))
        v = np.sum(raws, axis=0).astype(np.float32)
        q = float(np.clip((n - self.NORM_LO) / (self.NORM_HI - self.NORM_LO), 0.05, 1.0))
        return v / (np.linalg.norm(v) + 1e-9), q

    def embed(self, frame, kps) -> np.ndarray:
        return self.embed_q(frame, kps)[0]


_ALIASES = {"full": "fused", "gothamfaces": "fused", "lite": "arcface", "basic": "sface"}


def _choice() -> str:
    want = os.environ.get("GOTHAM_RECOGNISER", "").lower()
    if not want:
        try:
            from ..assistant import keys
            want = (keys.load().get("recogniser") or "").lower()
        except Exception:
            want = ""
    want = _ALIASES.get(want, want)
    if want == "sface" or not arcface.available():
        return "sface"
    if want == "arcface" or not adaface.available():
        return "arcface"
    return "fused"


def make_embedder():
    return {"fused": GothamFaces, "arcface": _ArcFaceAdapter, "sface": _SFaceAdapter}[_choice()]()


def recogniser_profile() -> RecogniserProfile:
    return {"fused": FUSED, "arcface": ARCFACE, "sface": SFACE}[_choice()]


def make_detector(prefer_gpu: bool = True):
    if scrfd.available():
        d = scrfd.SCRFDDetector(score_threshold=SCRFD.threshold, prefer_gpu=prefer_gpu)
        d.profile = SCRFD
    else:
        d = YuNetDetector(score_threshold=YUNET.threshold, prefer_gpu=prefer_gpu)
        d.profile = YUNET
    return d
