from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import MODELS
from .runtime import shared_session
from .yunet import YuNetDetector

SCRFD_MODEL = MODELS / "det_10g.onnx"
INPUT = 640
STRIDES = (8, 16, 32)
ANCHORS = 2


def available() -> bool:
    return SCRFD_MODEL.exists()


_CENTERS: dict = {}


def _centers(stride: int, h: int, w: int) -> np.ndarray:
    key = (stride, h, w)
    if key not in _CENTERS:
        ys, xs = np.mgrid[:h // stride, :w // stride]
        c = (np.stack([xs, ys], -1).reshape(-1, 2) * stride).astype(np.float32)
        _CENTERS[key] = np.repeat(c, ANCHORS, axis=0)
    return _CENTERS[key]


@dataclass
class SCRFDDetector(YuNetDetector):
    score_threshold: float = 0.5

    def __post_init__(self):
        self.session = shared_session(SCRFD_MODEL, self.prefer_gpu)
        outs = self.session.output_names
        self._idx = {s: (i, i + 3, i + 6) for i, s in enumerate(STRIDES)}
        assert len(outs) == 9, outs

    def _infer(self, img: np.ndarray, thresh: float):
        h, w = img.shape[:2]
        scale = INPUT / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        ch = cw = INPUT
        canvas = np.zeros((ch, cw, 3), np.uint8)
        canvas[:nh, :nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (cw, ch), (127.5, 127.5, 127.5), swapRB=True)
        out = self.session.run(blob)
        boxes, scores, kps = [], [], []
        for s in STRIDES:
            si, bi, ki = self._idx[s]
            sc = out[si].reshape(-1)
            keep = sc >= thresh
            if not keep.any():
                continue
            c = _centers(s, ch, cw)[keep]
            d = out[bi].reshape(-1, 4)[keep] * s
            x1, y1 = c[:, 0] - d[:, 0], c[:, 1] - d[:, 1]
            x2, y2 = c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]
            boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], 1))
            k = out[ki].reshape(-1, 5, 2)[keep] * s
            kps.append(k + c[:, None, :])
            scores.append(sc[keep])
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)
        inv = 1.0 / scale
        return np.concatenate(boxes) * inv, np.concatenate(scores), np.concatenate(kps) * inv
