from __future__ import annotations

from dataclasses import dataclass

from . import adaface, arcface


@dataclass(frozen=True)
class RecogniserProfile:
    name: str
    dim: int
    match: float
    margin: float
    new: float
    learn: float
    drop: float
    reacquire: float
    label: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    threshold: float
    verify_below: float
    memory_min: float
    enrol_min: float
    label: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name


FUSED = RecogniserProfile(f"fused:{arcface.NAME}+{adaface.NAME}", 1024, match=0.40, margin=0.075, new=0.26, learn=0.43,
                          drop=0.21, reacquire=0.38, label="GothamFaces")
ARCFACE = RecogniserProfile(arcface.NAME, 512, match=0.42, margin=0.08, new=0.28, learn=0.45, drop=0.22, reacquire=0.40,
                            label="GothamFaces Lite")
SFACE = RecogniserProfile("sface_2021dec", 128, match=0.52, margin=0.06, new=0.40, learn=0.55, drop=0.38, reacquire=0.50,
                          label="GothamFaces Basic")

SCRFD = DetectorProfile("scrfd_10g", threshold=0.45, verify_below=0.60, memory_min=0.60, enrol_min=0.70,
                        label="GothamFaces Detector")
YUNET = DetectorProfile("yunet_2023mar", threshold=0.60, verify_below=0.70, memory_min=0.70, enrol_min=0.80,
                        label="GothamFaces Detector Lite")


def display_name(key: str) -> str:
    for p in (FUSED, ARCFACE, SFACE):
        if p.name == key:
            return p.display
    return "previous model"
