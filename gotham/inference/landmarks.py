from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..config import LANDMARKER_MODEL

CROP = 256
CONTEXT = 1.7


@dataclass
class LandmarkResult:
    points: np.ndarray
    blendshapes: dict[str, float]
    matrix: np.ndarray | None
    crop_scale: float


_model_bytes: bytes | None = None


def _model() -> bytes:
    global _model_bytes
    if _model_bytes is None:
        _model_bytes = LANDMARKER_MODEL.read_bytes()
    return _model_bytes


class FaceLandmarks:
    def __init__(self, min_conf: float = 0.3):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_model()),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def close(self):
        self._lm.close()

    def __del__(self):
        try:
            self._lm.close()
        except Exception:
            pass

    @staticmethod
    def crop_transform(box: np.ndarray) -> np.ndarray:
        x, y, w, h = box
        side = max(w, h) * CONTEXT
        cx, cy = x + w / 2, y + h / 2
        s = CROP / side
        return np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)

    def run(self, frame_rgb: np.ndarray, box: np.ndarray) -> LandmarkResult | None:
        M = self.crop_transform(box)
        s = float(M[0, 0])
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        crop = cv2.warpAffine(frame_rgb, M, (CROP, CROP), flags=interp, borderMode=cv2.BORDER_CONSTANT)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        res = self._lm.detect(img)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        pts = np.array([(p.x, p.y, p.z) for p in lms], np.float32) * CROP
        inv = 1.0 / s
        out = np.empty_like(pts)
        out[:, 0] = (pts[:, 0] - M[0, 2]) * inv
        out[:, 1] = (pts[:, 1] - M[1, 2]) * inv
        out[:, 2] = pts[:, 2] * inv
        bs = {}
        if res.face_blendshapes:
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        mat = None
        if res.facial_transformation_matrixes:
            mat = np.array(res.facial_transformation_matrixes[0], np.float32)
        return LandmarkResult(out, bs, mat, inv)
