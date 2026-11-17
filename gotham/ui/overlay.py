from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections as _C
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from ..analysis import geometry as geo
from . import assets as A


def _pairs(conns) -> np.ndarray:
    return np.array([(c.start, c.end) for c in conns], np.int32)


TESSELATION = _pairs(_C.FACE_LANDMARKS_TESSELATION)
CONTOURS = _pairs(_C.FACE_LANDMARKS_CONTOURS)

LAYER_HELP = {
    "indicators": "reticles / boxes around faces",
    "labels": "subject tags and state",
    "trails": "movement trails and predicted trajectory",
    "web": "full face-mesh wireframe (tessellation, 'spiderweb')",
    "mesh": "468 landmark points",
    "contours": "face oval, brows, eyes, nose and lip outlines",
    "eyes": "eye outlines and iris rings",
    "gaze": "gaze direction rays",
    "pose": "head-pose axes",
    "quality": "image-quality bars",
    "grid": "crosshair grid over video",
    "telemetry": "feed telemetry text",
    "dossier": "Samaritan dossier panel beside focused and known subjects",
    "scan": "scanning sweep over faces being identified",
    "smooth": "video at the camera's full frame rate, overlays following each subject's motion between analysis "
              "frames (off: the picture steps with the analysis)",
}


@dataclass
class Layers:
    indicators: bool = True
    labels: bool = True
    trails: bool = True
    mesh: bool = False
    web: bool = False
    contours: bool = False
    eyes: bool = True
    gaze: bool = True
    pose: bool = True
    quality: bool = True
    grid: bool = True
    telemetry: bool = True
    dossier: bool = True
    scan: bool = True
    smooth: bool = True
    style: str = "samaritan"

    def toggle(self, name: str):
        if name == "style":
            self.style = "machine" if self.style == "samaritan" else "samaritan"
        elif hasattr(self, name):
            setattr(self, name, not getattr(self, name))

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Xf:
    s: float
    ox: float
    oy: float

    def p(self, x, y) -> QPointF:
        return QPointF(self.ox + x * self.s, self.oy + y * self.s)

    def r(self, box) -> QRectF:
        x, y, w, h = box
        return QRectF(self.ox + x * self.s, self.oy + y * self.s, w * self.s, h * self.s)


_METRICS: dict = {}


def metrics(font: QFont) -> QFontMetricsF:
    key = (font.family(), font.pointSizeF(), font.bold())
    fm = _METRICS.get(key)
    if fm is None:
        fm = _METRICS[key] = QFontMetricsF(font)
    return fm


def tag(p: QPainter, x: float, y: float, text: str, font, invert=True, red=False, align_right=False) -> QRectF:
    fm = metrics(font)
    w = fm.horizontalAdvance(text) + 6
    h = fm.height() + 1
    if align_right:
        x -= w
    r = QRectF(x, y, w, h)
    bg = A.RED if red else (A.BLACK if invert else A.WHITE)
    fg = A.WHITE if (invert or red) else A.BLACK
    p.fillRect(r, bg)
    p.setFont(font)
    p.setPen(fg)
    p.drawText(r.adjusted(3, 0, -3, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
    return r


def shadow_text(p: QPainter, pt: QPointF, text: str, font, color=A.WHITE, align=Qt.AlignLeft, width=600):
    p.setFont(font)
    fm = metrics(font)
    r = QRectF(pt.x() - (width if align & Qt.AlignRight else 0), pt.y(), width, fm.height())
    p.setPen(QColor(0, 0, 0, 170))
    p.drawText(r.translated(1.2, 1.2), align, text)
    p.setPen(color)
    p.drawText(r, align, text)


_GRID: dict = {}


def draw_crosshair_grid(p: QPainter, rect: QRectF, step: float = 90.0):
    key = (round(rect.left(), 1), round(rect.top(), 1), round(rect.width(), 1), round(rect.height(), 1), step)
    lines = _GRID.get(key)
    if lines is None:
        lines = []
        x0 = rect.left() + (rect.width() % step) / 2
        y0 = rect.top() + (rect.height() % step) / 2
        y = y0
        while y <= rect.bottom():
            x = x0
            while x <= rect.right():
                lines += [QLineF(x - 3, y, x + 3, y), QLineF(x, y - 3, x, y + 3)]
                x += step
            y += step
        if len(_GRID) > 32:
            _GRID.clear()
        _GRID[key] = lines
    p.setPen(QPen(QColor(255, 255, 255, 46), 1))
    p.drawLines(lines)


_SCALED: dict = {}


def _scaled(name: str, w: float, h: float, dpr: float) -> QPixmap | None:
    pw, ph = max(2, int(round(w * dpr / 2)) * 2), max(2, int(round(h * dpr / 2)) * 2)
    key = (name, pw, ph)
    pm = _SCALED.get(key)
    if pm is None:
        src = A.pixmap(name)
        if src.isNull():
            return None
        pm = src.scaled(pw, ph, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        if len(_SCALED) > 400:
            _SCALED.clear()
        _SCALED[key] = pm
    return pm


def _indicator(p: QPainter, name: str, centre: QPointF, size: float, opacity: float = 1.0):
    src = A.pixmap(name)
    if src.isNull():
        return
    w, h = size * src.width() / src.height(), size
    dev = p.device()
    pm = _scaled(name, w, h, dev.devicePixelRatioF() if dev is not None else 1.0)
    if pm is None:
        return
    p.setOpacity(opacity)
    p.drawPixmap(QPointF(centre.x() - w / 2, centre.y() - h / 2), pm)
    p.setOpacity(1.0)


def _quality_bar(p: QPainter, x, y, w, q):
    p.fillRect(QRectF(x, y, w, 3), QColor(0, 0, 0, 160))
    p.fillRect(QRectF(x, y, w * max(0.0, min(1.0, q)), 3), A.WHITE if q >= 0.45 else A.RED)


def draw_face(p: QPainter, xf: Xf, face, L: Layers, *, label: str | None = None, state: str | None = None,
              focused=False, coasting=False, box=None, sub: list[str] | None = None, blink=True):
    box = face.box if box is None else box
    r = xf.r(box)
    c = r.center()
    size = max(r.width(), r.height())
    small = size < 34
    if L.indicators:
        if L.style == "machine":
            name = "machine-indicator-yellow" if focused else "machine-indicator-white"
            _indicator(p, name, c, size * 1.25, 0.45 if coasting else 0.95)
        else:
            name = "samaritan-indicator3" if focused else ("samaritan-indicator" if coasting else "samaritan-indicator2")
            _indicator(p, name, c, size * 1.55, 0.55 if coasting else 0.95)
    if focused and blink:
        a = 0.35 + 0.65 * abs(math.sin(time.monotonic() * math.pi))
        sig = A.pixmap("samaritan-signal")
        if not sig.isNull():
            w = max(14.0, size * 0.22)
            h = w * sig.height() / sig.width()
            p.setOpacity(a)
            p.drawPixmap(QRectF(c.x() - w / 2, r.top() - size * 0.42 - h, w, h), sig, QRectF(sig.rect()))
            p.setOpacity(1.0)
    if face is not None and not coasting:
        if face.mesh is not None and not small:
            P = face.mesh
            if L.web or L.contours:
                X = xf.ox + P[:, 0] * xf.s
                Y = xf.oy + P[:, 1] * xf.s
                for on, edges, col, wdt in ((L.web, TESSELATION, QColor(255, 255, 255, 95), 0.7),
                                            (L.contours, CONTOURS, QColor(255, 255, 255, 220), 1.2)):
                    if on:
                        p.setPen(QPen(col, wdt))
                        a, b = edges[:, 0], edges[:, 1]
                        p.drawLines([QLineF(x0, y0, x1, y1) for x0, y0, x1, y1 in
                                     zip(X[a].tolist(), Y[a].tolist(), X[b].tolist(), Y[b].tolist())])
            if L.mesh:
                p.setPen(QPen(QColor(255, 255, 255, 120), max(1.0, xf.s * 0.9)))
                p.drawPoints(QPolygonF([xf.p(x, y) for x, y, _ in P[:468]]))
            if L.eyes:
                p.setPen(QPen(QColor(255, 255, 255, 220), 1))
                for contour in (geo.R_EYE_CONTOUR, geo.L_EYE_CONTOUR):
                    ex, ey = (xf.ox + P[contour, 0] * xf.s).tolist(), (xf.oy + P[contour, 1] * xf.s).tolist()
                    p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in zip(ex, ey)]))
                for e in (geo.R_EYE, geo.L_EYE):
                    ic = P[e["iris"], :2]
                    rad = float(np.linalg.norm(P[list(e["iris_ring"]), :2] - ic, axis=1).mean()) * xf.s
                    p.setPen(QPen(A.RED, 1.2))
                    p.drawEllipse(xf.p(*ic), rad, rad)
            if L.gaze and face.gaze is not None:
                g = face.gaze
                length = size * 0.9
                dx = math.sin(math.radians(g.yaw)) * length
                dy = -math.sin(math.radians(g.pitch)) * length
                pen = QPen(QColor(255, 255, 255, 230) if g.reliable else QColor(255, 255, 255, 90), 1.4)
                p.setPen(pen)
                for o in (g.origin_r, g.origin_l):
                    a = xf.p(*o)
                    p.drawLine(a, QPointF(a.x() + dx, a.y() + dy))
        if L.pose and face.pose is not None and not small:
            nose = face.mesh[geo.NOSE_TIP, :2] if face.mesh is not None else face.kps[2]
            o = xf.p(*nose)
            R = face.pose.R
            ln = size * 0.35
            for col, colour in ((0, A.WHITE), (1, A.GREY), (2, A.RED)):
                v = R[:, col] * (-1 if col == 2 else 1)
                p.setPen(QPen(colour, 1.6))
                p.drawLine(o, QPointF(o.x() + v[0] * ln, o.y() + v[1] * ln))
        elif face.mesh is None and L.eyes:
            p.setPen(QPen(A.WHITE, 1))
            for kx, ky in face.kps:
                p.drawEllipse(xf.p(kx, ky), 1.5, 1.5)
    if L.labels:
        fnt = A.mono(7.5)
        y = r.bottom() + size * (0.3 if L.style == "samaritan" else 0.15) + 2
        x = c.x() - size * 0.5
        lines = []
        if label:
            lines.append((label, "inv"))
        if state:
            lines.append((state, "red" if state in ("REACQUIRED", "NEW PROFILE") else "white"))
        for s in sub or []:
            lines.append((s, "inv"))
        for text, kind in lines:
            rr = tag(p, x, y, text, fnt, invert=kind == "inv", red=kind == "red")
            y = rr.bottom() + 1
        if L.quality and face is not None and face.quality and not coasting:
            _quality_bar(p, x, y + 1, max(40.0, size * 0.8), face.quality.get("overall", 0))


def draw_scan(p: QPainter, xf: Xf, box):
    r = xf.r(box)
    if r.height() < 14:
        return
    phase = (time.monotonic() * 1.6) % 1.0
    y = r.top() + phase * r.height()
    p.setPen(QPen(QColor(255, 0, 0, 200), 1.4))
    p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
    p.setPen(QPen(QColor(255, 0, 0, 60), 6))
    p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
    k = max(5.0, r.width() * 0.18)
    p.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
    for cx, cy, dx, dy in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                           (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
        p.drawLine(QPointF(cx, cy), QPointF(cx + dx * k, cy))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * k))


def draw_dossier(p: QPainter, xf: Xf, box, title: str, rows: list[tuple[str, str]], bounds: QRectF,
                 alert: bool = False):
    r = xf.r(box)
    fl, fv = A.mono(7.2), A.mono(7.2)
    fm = metrics(fl)
    lw = max((fm.horizontalAdvance(k) for k, _ in rows), default=40) + 10
    vw = max((fm.horizontalAdvance(v) for _, v in rows), default=40) + 10
    w = max(lw + vw + 14, fm.horizontalAdvance(title) + 24, 150)
    lh = fm.height() + 4
    h = lh * (len(rows) + 1) + 10
    size = max(r.width(), r.height())
    x = r.right() + size * 0.45 + 8
    if x + w > bounds.right() - 4:
        x = r.left() - size * 0.45 - 8 - w
    y = min(max(r.top() - lh, bounds.top() + 4), bounds.bottom() - h - 4)
    x = min(max(x, bounds.left() + 4), bounds.right() - w - 4)
    panel = QRectF(x, y, w, h)
    p.setPen(QPen(QColor(255, 255, 255, 140), 1))
    anchor = QPointF(panel.left() if panel.left() > r.center().x() else panel.right(), panel.top() + lh / 2)
    p.drawLine(QPointF(r.center().x() + (size * 0.5 if anchor.x() > r.center().x() else -size * 0.5), r.center().y()), anchor)
    p.fillRect(panel, QColor(0, 0, 0, 200))
    p.setPen(QPen(QColor(255, 255, 255, 230), 1))
    p.drawRect(panel)
    tb = QRectF(x, y, w, lh)
    p.fillRect(tb, A.RED if alert else A.WHITE)
    p.setFont(fl)
    p.setPen(A.WHITE if alert else A.BLACK)
    p.drawText(tb.adjusted(6, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, title)
    yy = y + lh + 4
    for k, v in rows:
        p.setPen(QColor(220, 220, 220))
        p.setFont(fl)
        p.drawText(QRectF(x + 6, yy, lw, lh - 2), Qt.AlignVCenter | Qt.AlignLeft, k)
        vb = QRectF(x + 6 + lw, yy + 1, fm.horizontalAdvance(v) + 8, lh - 4)
        p.fillRect(vb, QColor(96, 96, 96, 210))
        p.setPen(A.WHITE)
        p.setFont(fv)
        p.drawText(vb.adjusted(4, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, v)
        yy += lh


def ago(t: float) -> str:
    d = max(0.0, time.time() - t)
    if d < 90:
        return "JUST NOW"
    if d < 3600:
        return f"{d / 60:.0f}M AGO"
    if d < 86400 * 2:
        return f"{d / 3600:.0f}H AGO"
    return f"{d / 86400:.0f}D AGO"


_FADE_STEPS = 10


def draw_trail(p: QPainter, xf: Xf, trail, now: float, horizon: float, velocity=None, focused=False):
    if len(trail) < 2:
        return
    a = np.asarray(trail, np.float64)
    T, X, Y = a[:, 0], xf.ox + a[:, 1] * xf.s, xf.oy + a[:, 2] * xf.s
    dt = np.diff(T)
    span = max(40.0, xf.s * 200)
    ok = dt <= 0.8
    ok &= np.hypot(np.diff(X), np.diff(Y)) <= span * np.maximum(dt, 1 / 30) * 3
    alpha = np.maximum(0.05, 1 - (now - T[1:]) / horizon)
    level = np.minimum((alpha * _FADE_STEPS).astype(np.int32), _FADE_STEPS - 1)
    Xl, Yl = X.tolist(), Y.tolist()
    for lv in np.unique(level[ok]).tolist():
        idx = np.flatnonzero(ok & (level == lv)).tolist()
        al = int(255 * min(1.0, (lv + 1) / _FADE_STEPS))
        p.setPen(QPen(QColor(255, 0, 0, al) if focused else QColor(255, 255, 255, al), 1.6))
        p.drawLines([QLineF(Xl[i], Yl[i], Xl[i + 1], Yl[i + 1]) for i in idx])
    if velocity is not None and float(np.hypot(*velocity)) > 20:
        last = QPointF(Xl[-1], Yl[-1])
        pen = QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine)
        p.setPen(pen)
        vx, vy = velocity[0] * 0.6 * xf.s, velocity[1] * 0.6 * xf.s
        cap = 120.0 * max(xf.s, 0.3)
        n = math.hypot(vx, vy)
        if n > cap:
            vx, vy = vx * cap / n, vy * cap / n
        p.drawLine(last, QPointF(last.x() + vx, last.y() + vy))
