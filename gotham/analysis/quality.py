from __future__ import annotations

import cv2
import numpy as np


def _ramp(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def face_quality(gray: np.ndarray, box: np.ndarray, det_score: float, yaw: float | None,
                 pitch: float | None, has_mesh: bool, eye_open: tuple[float, float] | None) -> dict[str, float]:
    H, W = gray.shape[:2]
    x, y, w, h = box
    x0, y0 = int(max(0, x)), int(max(0, y))
    x1, y1 = int(min(W, x + w)), int(min(H, y + h))
    area = max(w * h, 1.0)
    inside = max(0, x1 - x0) * max(0, y1 - y0) / area
    q: dict[str, float] = {}
    if x1 - x0 < 4 or y1 - y0 < 4:
        roi = np.zeros((8, 8), np.uint8)
    else:
        roi = gray[y0:y1, x0:x1]
    norm = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(norm, cv2.CV_32F).var()
    mean = float(norm.mean())
    std = float(norm.std())
    q["resolution"] = _ramp(float(h), 20, 110)
    q["sharpness"] = _ramp(np.log10(lap + 1.0), 1.3, 2.6)
    q["exposure"] = float(np.clip(1.0 - abs(mean - 128) / 100.0, 0.0, 1.0))
    q["contrast"] = _ramp(std, 12, 50)
    if yaw is not None:
        q["frontality"] = float(np.clip(np.cos(np.radians(yaw)) * np.cos(np.radians(pitch or 0.0)), 0, 1)) ** 2
    else:
        q["frontality"] = 0.5
    vis = inside * (0.6 + 0.4 * (np.cos(np.radians(min(abs(yaw or 0), 90)))))
    if not has_mesh:
        vis *= 0.75
    if eye_open is not None:
        vis *= 0.85 + 0.15 * min(1.0, min(eye_open) / 0.2)
    q["visibility"] = float(np.clip(vis, 0, 1))
    q["detection"] = float(det_score)
    weights = dict(resolution=1.2, sharpness=1.0, exposure=0.7, contrast=0.5, frontality=0.8, visibility=1.2, detection=0.6)
    logsum = sum(wt * np.log(max(q[k], 0.02)) for k, wt in weights.items())
    q["overall"] = float(np.exp(logsum / sum(weights.values())))
    return q
