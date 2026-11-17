from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
LANDMARKER_MODEL = MODELS / "face_landmarker.task"
SFACE_MODEL = MODELS / "face_recognition_sface_2021dec.onnx"

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|timeout;5000000|fflags;nobuffer|flags;low_delay",
)
