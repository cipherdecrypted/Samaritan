from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget

from ..core.feed import Feed
from ..sources.capture import Status
from . import assets as A
from .overlay import (Layers, Xf, ago, draw_crosshair_grid, draw_dossier, draw_face, draw_scan, draw_trail,
                      shadow_text, tag)

PAGE_BG = QColor(244, 244, 244)


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    fmt = QImage.Format_Grayscale8 if arr.ndim == 2 else QImage.Format_BGR888
    return QImage(arr.data, w, h, arr.strides[0], fmt).copy()


def draw_frame(p: QPainter, frame: np.ndarray, target: QRectF):
    dev = p.device()
    dpr = dev.devicePixelRatioF() if dev is not None else 1.0
    tw = max(1, int(round(target.width() * dpr)))
    th = max(1, int(round(target.height() * dpr)))
    fh, fw = frame.shape[:2]
    if (tw, th) != (fw, fh):
        interp = cv2.INTER_AREA if tw < fw else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (tw, th), interpolation=interp)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    img = QImage(rgba.data, tw, th, rgba.strides[0], QImage.Format_RGB32)
    img.setDevicePixelRatio(dpr)
    p.drawImage(QPointF(target.left(), target.top()), img)
    del img, rgba


class FeedView(QWidget):
    selected = Signal(int)
    subjectClicked = Signal(int, int)
    expandToggled = Signal(int)
    menuAction = Signal(int, str)

    def __init__(self, feed: Feed, layers: Layers, hub, parent=None):
        super().__init__(parent)
        self.feed = feed
        self.layers = layers
        self.hub = hub
        self.is_selected = False
        self._seq: tuple = (None, None)
        self._painted = 0.0
        self._sprite_key = None
        self._sprites: list = []
        self._live_items: list = []
        self._xf: Xf | None = None
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setMinimumSize(160, 100)

    def refresh(self, min_interval: float = 0.0):
        r = self.feed.latest()
        live = self.feed.capture.latest() if (self.layers.smooth and r is not None) else None
        key = (r.seq if r else -1, live.seq if live is not None else -1)
        now = time.monotonic()
        focused = bool(self.hub.focus and self.hub.focus[0] == self.feed.id)
        if r is None or key[0] != self._seq[0] or focused or (key != self._seq and now - self._painted >= min_interval):
            self._seq = key
            self._painted = now
            self.update()

    @staticmethod
    def _shifts(r, xf: Xf, dt: float) -> dict:
        if dt <= 0:
            return {}
        d = min(dt, 0.15)
        return {t.id: QPointF(float(t.velocity[0]) * d * xf.s, float(t.velocity[1]) * d * xf.s)
                for t in r.tracks if not t.suppressed}

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), A.BLACK)
        r = self.feed.latest()
        st = self.feed.stats
        now = time.monotonic()
        L = self.layers
        if r is not None:
            frame, dt = r.frame, 0.0
            if L.smooth and not self.feed.capture.paused:
                live = self.feed.capture.latest()
                if live is not None and live.image.shape[:2] == r.frame.shape[:2] and 0.0 < live.t - r.t < 0.5:
                    frame, dt = live.image, live.t - r.t
            fh, fw = frame.shape[:2]
            s = min(W / fw, H / fh)
            ox, oy = (W - fw * s) / 2, (H - fh * s) / 2
            xf = Xf(s, ox, oy)
            self._xf = xf
            target = QRectF(ox, oy, fw * s, fh * s)
            draw_frame(p, frame, target)
            if L.grid:
                draw_crosshair_grid(p, target)
            if L.smooth:
                focus = self.hub.focus
                fsid = focus[1] if focus and focus[0] == self.feed.id else None
                dpr = self.devicePixelRatioF()
                key = (r.seq, W, H, dpr, tuple(L.as_dict().values()), fsid, self.feed.tracking,
                       int(now * 6) if fsid is not None else 0)
                if key != self._sprite_key:
                    clip = QRectF(-40, -40, W + 80, H + 80)
                    sprites, live_items = [], []
                    for tid, rect, fn in self._items(r, xf, target, now):
                        if rect is None:
                            live_items.append((tid, None, fn))
                            continue
                        sp = self._sprite(tid, rect.intersected(clip), dpr, fn)
                        if sp is not None:
                            sprites.append(sp)
                    self._sprites, self._live_items, self._sprite_key = sprites, live_items, key
                shift = self._shifts(r, xf, dt)
                for tid, origin, img in self._sprites:
                    o = shift.get(tid)
                    p.drawImage(origin + o if o is not None else origin, img)
                for tid, _, fn in self._live_items:
                    o = shift.get(tid)
                    p.save()
                    if o is not None:
                        p.translate(o)
                    fn(p)
                    p.restore()
            else:
                for _, _, fn in self._items(r, xf, target, now):
                    fn(p)
            if L.telemetry:
                self._telemetry(p, r, W, H)
        self._caption(p, W, H)
        if st.status != Status.LIVE or r is None:
            self._status_block(p, W, H, st)
        if self.is_selected:
            p.setPen(QPen(A.WHITE, 2))
            p.drawRect(QRectF(1, 1, W - 2, H - 2))
            p.setPen(QPen(A.RED, 2))
            p.drawLine(QPointF(1, 1), QPointF(24, 1))
        p.end()

    @staticmethod
    def _sprite(tid, rect: QRectF, dpr: float, fn):
        x0, y0 = math.floor(rect.left()), math.floor(rect.top())
        w, h = math.ceil(rect.right()) - x0, math.ceil(rect.bottom()) - y0
        if w < 1 or h < 1:
            return None
        img = QImage(int(math.ceil(w * dpr)), int(math.ceil(h * dpr)), QImage.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        img.fill(Qt.transparent)
        sp = QPainter(img)
        sp.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        sp.translate(-x0, -y0)
        fn(sp)
        sp.end()
        return tid, QPointF(x0, y0), img

    @staticmethod
    def _face_bounds(xf: Xf, box) -> QRectF:
        r = xf.r(box)
        c = r.center()
        size = max(r.width(), r.height(), 1.0)
        left = c.x() - 1.25 * size - 10
        right = max(c.x() + 1.25 * size, c.x() - 0.5 * size + 250) + 10
        top = c.y() - 1.35 * size - 36
        bottom = max(c.y() + 1.25 * size, r.bottom() + 0.3 * size + 84) + 10
        return QRectF(left, top, right - left, bottom - top)

    def _items(self, r, xf: Xf, target: QRectF, now: float) -> list:
        L = self.layers
        focus = self.hub.focus
        fsid = focus[1] if focus and focus[0] == self.feed.id else None
        items = []
        if not self.feed.tracking:
            for f in r.faces:
                items.append((None, self._face_bounds(xf, f.box),
                              lambda p, f=f: draw_face(p, xf, f, L, label=f"FACE {f.score:.2f}", sub=None)))
            return items
        if L.trails:
            trails = [t for t in r.tracks if not t.suppressed and len(t.trail) >= 2]
            if trails:
                pts = np.concatenate([np.asarray(t.trail, np.float64)[:, 1:] for t in trails])
                m = 125.0 * max(xf.s, 0.3) + 6
                lo, hi = pts.min(0) * xf.s, pts.max(0) * xf.s
                rect = QRectF(xf.ox + lo[0] - m, xf.oy + lo[1] - m, hi[0] - lo[0] + 2 * m, hi[1] - lo[1] + 2 * m)
                horizon = self.feed.tracker.cfg.trail_seconds

                def trails_fn(p, trails=trails):
                    for t in trails:
                        draw_trail(p, xf, t.trail, r.t, horizon, t.velocity if t.visible else None,
                                   focused=t.id == fsid)
                items.append((None, rect, trails_fn))
        tracked = set()
        for t in r.tracks:
            if t.suppressed or t.state == "lost":
                continue
            coast = not t.visible
            if t.visible:
                tracked.add(id(t.face))
            sub = []
            if not coast and t.face.gaze is not None and t.face.gaze.reliable \
                    and abs(t.face.gaze.yaw) < 12 and abs(t.face.gaze.pitch) < 12:
                sub.append("GAZE: CAMERA")
            state = "OCCLUDED" if coast else ("REACQUIRED" if t.reacquired and now - t.last_seen < 0.1
                                              and len(t.trail) < 20 else "TRACKING")
            label = f"SUBJECT {t.label}"
            if t.person_id is not None:
                label = f"{t.label} · {self.hub.person_name(t.person_id).upper()[:22]}"
                if not coast:
                    state = "NEW PROFILE" if t.memory_state == "new" else f"KNOWN {t.person_score:.2f}"
            elif t.memory_state in ("matching", "uncertain") and not coast:
                state = "IDENTIFYING"

            def face_fn(p, t=t, label=label, state=state, coast=coast, sub=sub):
                draw_face(p, xf, t.face, L, label=label, state=state, focused=t.id == fsid,
                          coasting=coast, box=t.box, sub=sub)
            items.append((t.id, self._face_bounds(xf, t.box), face_fn))
            if L.scan and state == "IDENTIFYING":
                items.append((t.id, None, lambda p, t=t: draw_scan(p, xf, t.box)))
        if L.dossier:
            items += self._dossier_items(xf, r.tracks, fsid, target, now)
        for f in r.faces:
            if id(f) not in tracked and f.track_id is None:
                items.append((None, self._face_bounds(xf, f.box),
                              lambda p, f=f: draw_face(p, xf, f, L, label=None, state=None,
                                                       sub=[f"ACQUIRING {f.score:.2f}"])))
        return items

    def _dossier_items(self, xf: Xf, tracks, fsid, bounds: QRectF, now: float) -> list:
        out = []
        live = [t for t in tracks if t.visible and not t.suppressed]
        for t in live:
            info = self.hub.person_info(t.person_id) if t.person_id is not None else None
            if not (t.id == fsid or (info and info["label"]) or (info and len(live) <= 3)):
                continue
            f = t.face
            g = f.gaze
            gaze = None
            if g is not None:
                gaze = "CAMERA" if g.reliable and abs(g.yaw) < 12 and abs(g.pitch) < 12 else f"{g.yaw:+.0f}/{g.pitch:+.0f}"
            el = int(now - t.first_seen)
            if info:
                title = (info["label"] or info["code"]).upper()[:24]
                rows = [("PROFILE", info["code"]), ("SUBJECT", t.label),
                        ("MATCH", "NEW PROFILE" if t.memory_state == "new" else f"{t.person_score:.2f}"),
                        ("SIGHTINGS", str(max(info["sightings"], 1))), ("FIRST SEEN", ago(info["first_seen"])),
                        ("VIEWS", str(info["views"]))]
            else:
                title = f"SUBJECT {t.label}"
                rows = [("STATUS", "IDENTIFYING" if t.memory_state in ("matching", "uncertain") else "TRACKING"),
                        ("IN SESSION", f"{el // 60:02d}:{el % 60:02d}"), ("DETECTION", f"{f.score:.2f}")]
            if gaze:
                rows.append(("GAZE", gaze))
            if f.pose is not None:
                rows.append(("HEAD", f"Y{f.pose.yaw:+.0f} P{f.pose.pitch:+.0f}"))
            rows.append(("QUALITY", f"{f.quality.get('overall', 0):.2f}"))
            rr = xf.r(t.box)
            size = max(rr.width(), rr.height())
            top = min(rr.top() - 40, bounds.bottom() - 300)
            area = QRectF(rr.left() - 0.45 * size - 360, top, rr.width() + 0.9 * size + 720,
                          max(rr.bottom() + 10, top + 340) - top)
            out.append((t.id, area.intersected(bounds.adjusted(-2, -2, 2, 2)),
                        lambda p, t=t, title=title, rows=rows: draw_dossier(p, xf, t.box, title, rows, bounds,
                                                                            alert=t.id == fsid)))
        return out

    def _caption(self, p: QPainter, W, H):
        big = A.mono(max(11.0, min(20.0, W / 48)))
        sm = A.mono(max(7.0, min(10.0, W / 90)))
        x = W - 14
        shadow_text(p, QPointF(x, 10), self.feed.name.replace("-", " "), big, align=Qt.AlignRight)
        shadow_text(p, QPointF(x, 12 + big.pointSizeF() * 1.6), self.feed.spec.label.upper()[:40], sm,
                    align=Qt.AlignRight)
        shadow_text(p, QPointF(x, 14 + big.pointSizeF() * 1.6 + sm.pointSizeF() * 1.7),
                    time.strftime("%H:%M:%S"), big, align=Qt.AlignRight)
        if self.feed.tracking:
            sess = self.feed.tracker.started_at
            el = int(time.monotonic() - sess) if sess else 0
            tag(p, 12, 12, f"TRACKING.SESSION  {el // 60:02d}:{el % 60:02d}", sm, invert=False)
        if self.feed.capture.paused:
            tag(p, 12, 32, "PAUSED", sm, red=True)

    def _telemetry(self, p: QPainter, r, W, H):
        st = self.feed.stats
        f = A.mono(7)
        lines = [f"TK_{st.fps:05.1f}.CAP", f"TK_{r.process_fps:05.1f}.ANL", f"LAT_{r.latency_ms:04.0f}MS",
                 f"DET_{len(r.faces):03d}  TRK_{sum(1 for t in r.tracks if t.visible):03d}",
                 f"{st.width}x{st.height}  {'GPU' if 'Dml' in self.feed.provider else 'CPU'}"]
        y = H - 12 - len(lines) * 12
        for ln in lines:
            shadow_text(p, QPointF(W - 14, y), ln, f, color=QColor(255, 255, 255, 210), align=Qt.AlignRight)
            y += 12
        if self.feed.is_file and st.position:
            p.fillRect(QRectF(0, H - 2, W * st.position, 2), A.RED)

    def _status_block(self, p: QPainter, W, H, st):
        text = {Status.CONNECTING: "CONNECTING", Status.RECONNECTING: "SIGNAL LOST", Status.FAILED: "SOURCE UNAVAILABLE",
                Status.ENDED: "END OF FEED", Status.PAUSED: "PAUSED", Status.STOPPED: "OFFLINE"}.get(st.status, "")
        if not text:
            text = "AWAITING SIGNAL"
        f = A.mono(max(10.0, min(16.0, W / 50)))
        blink = "_" if int(time.monotonic() * 2) % 2 else " "
        r = tag(p, W / 2 - 100, H / 2 - 14, f"{text} {blink}", f, invert=False)
        if st.reconnects:
            tag(p, r.left(), r.bottom() + 4, f"RECONNECT ATTEMPTS: {st.reconnects}", A.mono(7.5))
        if st.error:
            tag(p, r.left(), r.bottom() + 22, st.error.upper(), A.mono(7.5), red=True)

    def _subject_at(self, pos) -> int | None:
        r = self.feed.latest()
        if r is None or self._xf is None:
            return None
        best = None
        for t in r.tracks:
            if t.suppressed or t.state == "lost":
                continue
            rr = self._xf.r(t.box).adjusted(-20, -20, 20, 40)
            if rr.contains(pos):
                d = (rr.center() - pos).manhattanLength()
                if best is None or d < best[0]:
                    best = (d, t.id)
        return best[1] if best else None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.selected.emit(self.feed.id)
            sid = self._subject_at(e.position())
            if sid is not None:
                self.subjectClicked.emit(self.feed.id, sid)

    def mouseDoubleClickEvent(self, e):
        self.expandToggled.emit(self.feed.id)

    def contextMenuEvent(self, e):
        self.selected.emit(self.feed.id)
        m = QMenu(self)
        m.setStyleSheet("QMenu{background:#000;color:#fff;border:2px solid #fff;font-family:'%s';font-size:9pt}"
                        "QMenu::item{padding:4px 18px}QMenu::item:selected{background:#fff;color:#000}"
                        % A.mono(9).family())
        f = self.feed
        acts = [("tracking", "STOP TRACKING SESSION" if f.tracking else "START TRACKING SESSION"),
                ("pause", "RESUME" if f.capture.paused else "PAUSE"),
                ("small", ("✓ " if f.settings.small_faces else "") + "SMALL-FACE MODE"),
                ("landmarks", ("✓ " if f.settings.landmarks else "") + "DENSE LANDMARKS"),
                ("expand", "EXPAND / RESTORE"), ("fullscreen", "FULL SCREEN  (ESC TO EXIT)"),
                ("close", "DISCONNECT FEED")]
        sid = self._subject_at(e.pos())
        if sid is not None:
            acts.insert(0, (f"focus:{sid}", f"FOCUS S{sid:02d}"))
            acts.insert(1, (f"suppress:{sid}", f"EXCLUDE S{sid:02d} FROM SESSION"))
        for key, text in acts:
            a = QAction(text, m)
            a.triggered.connect(lambda _=False, k=key: self.menuAction.emit(self.feed.id, k))
            m.addAction(a)
        m.exec(e.globalPos())


class FeedGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.views: list[FeedView] = []
        self.expanded: int | None = None
        self.gap = 8
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def add(self, v: FeedView):
        v.setParent(self)
        v.expandToggled.connect(self.toggle_expand)
        self.views.append(v)
        v.show()
        self._layout()

    def remove(self, feed_id: int):
        for v in list(self.views):
            if v.feed.id == feed_id:
                self.views.remove(v)
                v.setParent(None)
                v.deleteLater()
        if self.expanded == feed_id:
            self.expanded = None
        self._layout()

    def toggle_expand(self, fid: int):
        self.expanded = None if self.expanded == fid else fid
        self._layout()

    def set_selected(self, fid: int | None):
        for v in self.views:
            v.is_selected = v.feed.id == fid
            v.update()

    def refresh(self):
        vis = [v for v in self.views if v.isVisible()]
        gap = 0.0 if len(vis) <= 2 else 0.03
        for v in vis:
            v.refresh(gap)

    def resizeEvent(self, e):
        self._layout()

    def _layout(self):
        vs = self.views
        if self.expanded is not None and any(v.feed.id == self.expanded for v in vs):
            for v in vs:
                if v.feed.id == self.expanded:
                    v.setGeometry(0, 0, self.width(), self.height())
                    v.show()
                else:
                    v.hide()
            return
        n = len(vs)
        if not n:
            self.update()
            return
        W, H, g = self.width(), self.height(), self.gap
        best = None
        for cols in range(1, n + 1):
            rows = math.ceil(n / cols)
            tw = (W - g * (cols - 1)) / cols
            th = (H - g * (rows - 1)) / rows
            area = min(tw, th * 16 / 9) * min(th, tw * 9 / 16)
            if best is None or area > best[0]:
                best = (area, cols, rows, tw, th)
        _, cols, rows, tw, th = best
        for i, v in enumerate(vs):
            r, c = divmod(i, cols)
            v.setGeometry(int(c * (tw + g)), int(r * (th + g)), int(tw), int(th))
            v.show()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), PAGE_BG)
        if not self.views:
            p.setRenderHint(QPainter.TextAntialiasing)
            f = A.mono(14)
            p.setFont(f)
            p.setPen(A.BLACK)
            p.drawText(self.rect(), Qt.AlignCenter, "NO ACTIVE FEEDS\n\nSELECT A SOURCE")
        p.end()
