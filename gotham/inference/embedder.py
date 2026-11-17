from __future__ import annotations

import cv2
import numpy as np

from ..config import SFACE_MODEL
from .runtime import shared_session

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def align112(frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(kps[:3].astype(np.float32), TEMPLATE[:3])
    return cv2.warpAffine(frame_bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class AppearanceEmbedder:
    def __init__(self, prefer_gpu: bool = True):
        self.session = shared_session(SFACE_MODEL, prefer_gpu)

    def embed(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop = align112(frame_bgr, kps)
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32)
        v = self.session.run(blob)[0].reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)
