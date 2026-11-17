from __future__ import annotations

import cv2
import numpy as np

from . import adaface, arcface
from .embedder import align112
from .profiles import FUSED
from .runtime import run_batched, shared_session


class GothamFaces:
    profile = FUSED
    NORM_LO, NORM_HI = 16.0, 22.0
    FLIP_MAX = 4

    def __init__(self):
        a = arcface.ARCFACE_BATCH_FP16 if arcface.ARCFACE_BATCH_FP16.exists() else arcface.ARCFACE_BATCH_MODEL
        self._a = shared_session(a)
        self._b = shared_session(adaface.model_path())
        self.session = self._a
        self._exact: dict = {}

    def _finite(self, out: np.ndarray, blob: np.ndarray, which: str) -> np.ndarray:
        bad = ~np.isfinite(out).all(axis=1)
        if bad.any():
            path = arcface.ARCFACE_BATCH_MODEL if which == "a" else adaface.MODEL
            if which not in self._exact:
                self._exact[which] = shared_session(path) if path.exists() else None
            if self._exact[which] is not None:
                out[bad] = run_batched(self._exact[which], blob[bad])
        return out

    @staticmethod
    def _blob(crops: list[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32)
        return ((x - 127.5) / 127.5).transpose(0, 3, 1, 2).copy()

    def embed_crops(self, crops: list[np.ndarray], flip: bool | None = None) -> list[tuple[np.ndarray, float]]:
        n = len(crops)
        if not n:
            return []
        flip = n <= self.FLIP_MAX if flip is None else flip
        blob = self._blob(crops + ([c[:, ::-1].copy() for c in crops] if flip else []))
        A = self._finite(run_batched(self._a, blob), blob, "a")
        B = self._finite(run_batched(self._b, blob), blob, "b")
        if flip:
            a, b = A[:n] + A[n:], B[:n] + B[n:]
            norms = (np.linalg.norm(A[:n], axis=1) + np.linalg.norm(A[n:], axis=1)) / 2
        else:
            a, b, norms = A, B, np.linalg.norm(A, axis=1)
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
        b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
        v = np.concatenate([a, b], axis=1) / np.sqrt(2.0)
        q = np.clip((norms - self.NORM_LO) / (self.NORM_HI - self.NORM_LO), 0.05, 1.0)
        return [(v[i].astype(np.float32), float(q[i])) for i in range(n)]

    def embed_q_many(self, frame, kps_list) -> list[tuple[np.ndarray, float]]:
        return self.embed_crops([align112(frame, k) for k in kps_list]) if kps_list else []

    def embed_q(self, frame, kps) -> tuple[np.ndarray, float]:
        return self.embed_crops([align112(frame, kps)], flip=True)[0]

    def embed(self, frame, kps) -> np.ndarray:
        return self.embed_q(frame, kps)[0]
