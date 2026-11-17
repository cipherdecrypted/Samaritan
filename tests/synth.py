from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).parent / "data"


def face_patches() -> list[np.ndarray]:
    lena = cv2.imread(str(DATA / "lena.jpg"))
    a = lena[180:420, 190:400]
    sol = cv2.imread(str(DATA / "solvay.jpg"))
    b = cv2.resize(sol[1080:1260, 2240:2400], (180, 200), interpolation=cv2.INTER_CUBIC)
    return [a, b]


def tracking_sequence(n: int = 210, size=(1280, 720), seed: int = 3):
    W, H = size
    rng = np.random.default_rng(seed)
    bg = (rng.integers(20, 60, (H, W, 3))).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    A, B = face_patches()
    for i in range(n):
        f = bg.copy()
        t = i / 30.0
        gt = {}
        if i < 110:
            ax = int(60 + i * 11)
        elif i >= 150:
            ax = int(-150 + (i - 150) * 9)
        else:
            ax = None
        if ax is not None:
            ay = 120 + int(20 * np.sin(i / 15))
            _paste(f, A, ax, ay)
            gt["A"] = (ax, ay)
        bx, by = 900 + int(40 * np.sin(i / 25)), 430
        _paste(f, B, bx, by)
        gt["B"] = (bx, by)
        if 60 <= i < 75:
            cv2.rectangle(f, (ax - 30, 0), (ax + A.shape[1] + 30, H - 330), (35, 35, 35), -1)
            gt["A_occluded"] = True
        yield t, f, gt


def _paste(dst, src, x, y):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def crossing_sequence(n: int = 120, size=(1280, 720), seed: int = 5):
    W, H = size
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(20, 60, (H, W, 3)).astype(np.uint8), (0, 0), 3)
    A, B = face_patches()
    for i in range(n):
        f = bg.copy()
        ax = int(80 + i * (W - 400) / n)
        bx = int(W - 330 - i * (W - 400) / n)
        _paste(f, A, ax, 200)
        _paste(f, B, bx, 215)
        yield i / 30.0, f, {"A": (ax, 200), "B": (bx, 215)}


def meet_and_return_sequence(n: int = 150, size=(1280, 720), seed: int = 6):
    W, H = size
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(20, 60, (H, W, 3)).astype(np.uint8), (0, 0), 3)
    A, B = face_patches()
    cx = W // 2 - 105
    for i in range(n):
        f = bg.copy()
        if i < 50:
            p = i / 50
        elif i < 90:
            p = 1.0
        else:
            p = 1.0 - (i - 90) / 60
        jit = int(14 * np.sin(i / 2.0)) if 50 <= i < 90 else 0
        ax = int(60 + p * (cx - 60 - 120)) + jit
        bx = int(W - 270 - p * (W - 270 - cx - 120)) - jit
        _paste(f, A, ax, 220)
        _paste(f, B, bx, 230)
        yield i / 30.0, f, {"A": (ax, 220), "B": (bx, 230)}
