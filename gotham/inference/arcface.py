from __future__ import annotations

import cv2
import numpy as np

from ..config import MODELS
from .embedder import align112
from .runtime import shared_session

ARCFACE_MODEL = MODELS / "w600k_r50.onnx"
ARCFACE_BATCH_MODEL = MODELS / "w600k_r50_batch.onnx"
ARCFACE_BATCH_FP16 = MODELS / "w600k_r50_batch_fp16.onnx"
NAME = "arcface_w600k_r50"


def available() -> bool:
    return ARCFACE_MODEL.exists()


class ArcFaceEmbedder:
    dim = 512

    def __init__(self, prefer_gpu: bool = True):
        self.session = shared_session(ARCFACE_MODEL, prefer_gpu)

    @staticmethod
    def _blob(crops: list[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32)
        return ((x - 127.5) / 127.5).transpose(0, 3, 1, 2).copy()

    def embed_raw(self, frame_bgr: np.ndarray, kps_list: list[np.ndarray]) -> np.ndarray:
        if not kps_list:
            return np.zeros((0, self.dim), np.float32)
        out = [self.session.run(self._blob([align112(frame_bgr, k)]))[0][0] for k in kps_list]
        return np.stack(out).astype(np.float32)

    def embed_many(self, frame_bgr: np.ndarray, kps_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        raw = self.embed_raw(frame_bgr, kps_list)
        norms = np.linalg.norm(raw, axis=1)
        return raw / (norms[:, None] + 1e-9), norms

    def embed(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        return self.embed_many(frame_bgr, [kps])[0][0]
