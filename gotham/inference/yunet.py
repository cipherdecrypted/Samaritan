from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import YUNET_MODEL
from .runtime import Session, shared_session

INPUT = 640
STRIDES = (8, 16, 32)


@dataclass
class Detection:
    box: np.ndarray
    score: float
    kps: np.ndarray
    truncated: bool = False

    @property
    def xyxy(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x, y, x + w, y + h], np.float32)


def _grid(stride: int) -> tuple[np.ndarray, np.ndarray]:
    n = INPUT // stride
    rows, cols = np.mgrid[0:n, 0:n]
    return cols.reshape(-1).astype(np.float32), rows.reshape(-1).astype(np.float32)


_GRIDS = {s: _grid(s) for s in STRIDES}


@dataclass
class YuNetDetector:
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    prefer_gpu: bool = True
    session: Session = field(init=False)

    def __post_init__(self):
        self.session = shared_session(YUNET_MODEL, self.prefer_gpu)
        names = self.session.output_names
        self._order = {n: i for i, n in enumerate(names)}

    @property
    def provider(self) -> str:
        return self.session.provider

    def _infer(self, img: np.ndarray, thresh: float):
        h, w = img.shape[:2]
        scale = INPUT / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        canvas = np.zeros((INPUT, INPUT, 3), np.uint8)
        canvas[:nh, :nw] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(blob)
        o = self._order
        boxes, scores, kps = [], [], []
        for s in STRIDES:
            cls = np.clip(out[o[f"cls_{s}"]][0, :, 0], 0, 1)
            obj = np.clip(out[o[f"obj_{s}"]][0, :, 0], 0, 1)
            sc = np.sqrt(cls * obj)
            keep = sc >= thresh
            if not keep.any():
                continue
            cols, rows = _GRIDS[s]
            c, r = cols[keep], rows[keep]
            bb = out[o[f"bbox_{s}"]][0][keep]
            kp = out[o[f"kps_{s}"]][0][keep].reshape(-1, 5, 2)
            cx = (c + bb[:, 0]) * s
            cy = (r + bb[:, 1]) * s
            bw = np.exp(bb[:, 2]) * s
            bh = np.exp(bb[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1))
            k = np.empty_like(kp)
            k[..., 0] = (kp[..., 0] + c[:, None]) * s
            k[..., 1] = (kp[..., 1] + r[:, None]) * s
            kps.append(k)
            scores.append(sc[keep])
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)
        inv = 1.0 / scale
        return (np.concatenate(boxes) * inv, np.concatenate(scores), np.concatenate(kps) * inv)

    def detect(self, frame: np.ndarray, tiled: bool | str = False, threshold: float | None = None) -> list[Detection]:
        thresh = self.score_threshold if threshold is None else threshold
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        H, W = frame.shape[:2]
        B, S, K = [], [], []
        b, s, k = self._infer(frame, thresh)
        B.append(b), S.append(s), K.append(k)
        if tiled:
            tiles = list(self._tiles(W, H))
            if tiled == "auto":
                self._rr = (getattr(self, "_rr", -1) + 1) % len(tiles)
                tiles = [tiles[self._rr]]
            for (x0, y0, x1, y1) in tiles:
                b, s, k = self._infer(frame[y0:y1, x0:x1], thresh)
                if len(s):
                    b[:, 0] += x0
                    b[:, 1] += y0
                    k[..., 0] += x0
                    k[..., 1] += y0
                    B.append(b), S.append(s), K.append(k)
        boxes = np.concatenate(B)
        scores = np.concatenate(S)
        kps = np.concatenate(K)
        if not len(scores):
            return []
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), thresh, self.nms_threshold)
        idx = np.array(idx).reshape(-1)
        if tiled and len(idx) > 1:
            idx = self._suppress_contained(boxes, scores, idx)
        dets = []
        margin = 2.0
        for i in idx:
            x, y, w, h = boxes[i]
            trunc = x < margin or y < margin or x + w > W - margin or y + h > H - margin
            dets.append(Detection(boxes[i].astype(np.float32), float(scores[i]), kps[i].astype(np.float32), bool(trunc)))
        dets.sort(key=lambda d: -d.score)
        return dets

    @staticmethod
    def _suppress_contained(boxes, scores, idx, frac: float = 0.5):
        order = sorted(idx, key=lambda i: -scores[i])
        keep = []
        for i in order:
            x, y, w, h = boxes[i]
            ok = True
            for j in keep:
                X, Y, W2, H2 = boxes[j]
                ix = max(0.0, min(x + w, X + W2) - max(x, X))
                iy = max(0.0, min(y + h, Y + H2) - max(y, Y))
                if ix * iy > frac * min(w * h, W2 * H2):
                    ok = False
                    break
            if ok:
                keep.append(i)
        return np.array(keep)

    @staticmethod
    def _tiles(W: int, H: int, overlap: float = 0.25):
        nx = 3 if W / max(H, 1) > 2.0 else 2
        ny = 2
        tw = int(W / (nx - (nx - 1) * overlap))
        th = int(H / (ny - (ny - 1) * overlap))
        sx = (W - tw) / max(nx - 1, 1)
        sy = (H - th) / max(ny - 1, 1)
        for j in range(ny):
            for i in range(nx):
                x0, y0 = int(i * sx), int(j * sy)
                yield x0, y0, min(W, x0 + tw), min(H, y0 + th)
