from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)
ort.set_default_logger_severity(3)

_DML = "DmlExecutionProvider"
_CPU = "CPUExecutionProvider"
_GPU_LOCK = threading.Lock()
_KEEP_ALIVE: list = []


class Session:
    def __init__(self, path: Path, prefer_gpu: bool = True):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.provider = _CPU
        self.sess = None
        errors = []
        if prefer_gpu and _DML in ort.get_available_providers():
            try:
                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.sess = ort.InferenceSession(str(self.path), so, providers=[_DML, _CPU])
                self.provider = self.sess.get_providers()[0]
            except Exception as e:
                errors.append(repr(e))
                self.sess = None
        if self.sess is None:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 4
            self.sess = ort.InferenceSession(str(self.path), so, providers=[_CPU])
            self.provider = _CPU
            if errors:
                log.warning("GPU session failed for %s: %s", self.path.name, errors)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.output_names = [o.name for o in self.sess.get_outputs()]
        if self.provider == _DML:
            self._lock = _GPU_LOCK
            _KEEP_ALIVE.append(self.sess)

    @property
    def on_gpu(self) -> bool:
        return self.provider == _DML

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        with self._lock:
            return self.sess.run(self.output_names, {self.input_name: x})


_cache: dict[tuple[str, bool], Session] = {}
_cache_lock = threading.Lock()


def shared_session(path: Path, prefer_gpu: bool = True) -> Session:
    key = (str(path), prefer_gpu)
    with _cache_lock:
        if key not in _cache:
            _cache[key] = Session(path, prefer_gpu)
        return _cache[key]


def run_batched(session, blob: np.ndarray, size: int = 8) -> np.ndarray:
    out = []
    for i in range(0, len(blob), size):
        b = blob[i:i + size]
        n = len(b)
        if n < size:
            b = np.concatenate([b, np.zeros((size - n,) + b.shape[1:], np.float32)])
        out.append(session.run(b)[0][:n])
    return np.concatenate(out).astype(np.float32)
