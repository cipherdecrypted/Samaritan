from __future__ import annotations

from ..config import MODELS

MODEL = MODELS / "adaface_ir101_webface12m.onnx"
MODEL_FP16 = MODELS / "adaface_ir101_webface12m_fp16.onnx"
NAME = "adaface_ir101_webface12m"


def available() -> bool:
    return MODEL_FP16.exists() or MODEL.exists()


def model_path():
    return MODEL_FP16 if MODEL_FP16.exists() else MODEL
