from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

from ..analysis import geometry as geo
from ..analysis.pipeline import AnalysisSettings, Face, FaceAnalyzer
from ..inference.embedder import TEMPLATE
from . import assets as A
from .overlay import TESSELATION, Layers, Xf, draw_face, tag
from .viewport import PAGE_BG, draw_frame

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_SIDE = 2600


@dataclass
class ImageItem:
    path: Path
    image: np.ndarray | None = None
    faces: list[Face] = field(default_factory=list)
    pending: bool = True
    error: str = ""
    embs: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name


def png_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def jpg_data_url(img: np.ndarray, q=88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def square_crop(img: np.ndarray, box, out: int, context: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = box
    side = max(w, h) * context
    s = out / side
    M = np.array([[s, 0, out / 2 - s * (x + w / 2)], [0, s, out / 2 - s * (y + h / 2)]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT), M


def aligned_crop(img: np.ndarray, kps: np.ndarray, out: int = 192) -> tuple[np.ndarray, np.ndarray]:
    dst = TEMPLATE * (out / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(kps[:3].astype(np.float32), dst[:3].astype(np.float32))
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT), M


def draw_mesh_np(img: np.ndarray, pts2d: np.ndarray, colour=(255, 255, 255), web=True):
    if web:
        for a, b in TESSELATION:
            cv2.line(img, tuple(np.int32(pts2d[a] * 4)), tuple(np.int32(pts2d[b] * 4)), colour, 1, cv2.LINE_AA, shift=2)
    for e in (geo.R_EYE, geo.L_EYE):
        c = pts2d[e["iris"]]
        r = np.mean([np.linalg.norm(pts2d[i] - c) for i in e["iris_ring"]])
        cv2.circle(img, tuple(np.int32(c * 4)), int(r * 4), (0, 0, 255), 1, cv2.LINE_AA, shift=2)


class ImageLab:
    def __init__(self, on_change, memory=None):
        self.memory = memory
        self.items: list[ImageItem] = []
        self.current = 0
        self.ab: dict[str, str] = {}
        self.inspected: str | None = None
        self.on_change = on_change
        self._lock = threading.Lock()
        self._queue: list[ImageItem] = []
        self._worker: threading.Thread | None = None
        self.settings = AnalysisSettings(threshold=None, small_faces=True, landmarks=True, max_landmark_faces=64,
                                         min_landmark_size=16)
        self._cache: dict = {}

    def add_paths(self, paths):
        new = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                new += [ImageItem(q) for q in sorted(p.iterdir()) if q.suffix.lower() in IMAGE_EXT]
            elif p.suffix.lower() in IMAGE_EXT:
                new.append(ImageItem(p))
        if not new:
            return 0
        with self._lock:
            first_new = len(self.items)
            self.items += new
            self._queue += new
        self.current = first_new
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, daemon=True, name="imagelab")
            self._worker.start()
        self.on_change()
        return len(new)

    def clear(self):
        with self._lock:
            self.items, self._queue = [], []
        self.ab, self.inspected, self.current = {}, None, 0
        self._cache.clear()
        self.on_change()

    def _work(self):
        analyzer = FaceAnalyzer()
        from ..inference.models import make_embedder
        embedder = make_embedder()
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        return
                    it = self._queue.pop(0)
                try:
                    data = np.fromfile(str(it.path), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("unreadable image")
                    s = MAX_SIDE / max(img.shape[:2])
                    if s < 1:
                        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    it.image = img
                    it.faces = analyzer.analyze(img, self.settings).faces
                    it.embs = []
                    for f in it.faces:
                        v, f.recog_q = embedder.embed_q(img, f.kps)
                        it.embs.append(v)
                except Exception as e:
                    it.error = str(e)
                it.pending = False
                self.on_change()
        finally:
            analyzer.close()

    def face(self, key: str) -> tuple[ImageItem, Face] | None:
        try:
            i, j = (int(v) for v in key.split(":"))
            it = self.items[i]
            return it, it.faces[j]
        except (ValueError, IndexError):
            return None

    def set_ab(self, key: str):
        if self.ab.get("A") == key:
            self.ab.pop("A")
        elif self.ab.get("B") == key:
            self.ab.pop("B")
        elif "A" not in self.ab:
            self.ab["A"] = key
        else:
            self.ab["B"] = key
        self._cache.pop("compare", None)
        self.on_change()

    def inspect(self, key: str):
        self.inspected = key
        r = self.face(key)
        if r:
            self.current = self.items.index(r[0])
        self.on_change()

    def payload(self) -> tuple[list, dict]:
        imgs = [{"idx": i, "name": it.name, "faces": len(it.faces), "pending": it.pending, "sel": i == self.current}
                for i, it in enumerate(self.items)]
        lab: dict = {"detections": []}
        if self.items and 0 <= self.current < len(self.items):
            it = self.items[self.current]
            if it.image is not None:
                for j, f in enumerate(it.faces):
                    key = f"{self.current}:{j}"
                    ck = ("thumb", key)
                    if ck not in self._cache:
                        self._cache[ck] = jpg_data_url(square_crop(it.image, f.box, 96)[0])
                    ab = next((k for k, v in self.ab.items() if v == key), "")
                    lab["detections"].append({
                        "key": key, "thumb": self._cache[ck], "title": f"D{j + 1:02d}", "ab": ab,
                        "score": f"{f.score:.2f}", "size": int(f.box[3]),
                        "yaw": f"{f.pose.yaw:+.0f}" if f.pose else "--",
                        "q": f"{f.quality.get('overall', 0):.2f}", "qv": f.quality.get("overall", 0),
                        "match": self._match_text(it, j)})
        if self.inspected and self.face(self.inspected):
            lab["detail"] = self._detail(self.inspected)
        if "A" in self.ab and "B" in self.ab and self.face(self.ab["A"]) and self.face(self.ab["B"]):
            if "compare" not in self._cache:
                self._cache["compare"] = self._compare(self.ab["A"], self.ab["B"])
            lab["compare"] = self._cache["compare"]
        return imgs, lab

    def _best(self, it: ImageItem, j: int):
        if self.memory is None or j >= len(it.embs):
            return None
        ranked = self.memory.match(it.embs[j][None])
        return ranked[0] if ranked else None

    def _match_text(self, it: ImageItem, j: int) -> str:
        th = self.memory.th
        MATCH, NEW = th.match, th.new
        b = self._best(it, j)
        if not b or b[1] < NEW:
            return ""
        p = self.memory.person(b[0])
        name = p.name if p else f"P-{b[0]:04d}"
        return f"{'MATCH' if b[1] >= MATCH else 'POSSIBLE'} {name} {b[1]:.2f}"

    def _search(self, it: ImageItem, f) -> list[dict]:
        if self.memory is None:
            return []
        j = it.faces.index(f)
        if j >= len(it.embs):
            return []
        th = self.memory.th
        out = []
        for pid, score in self.memory.match(it.embs[j][None])[:3]:
            p = self.memory.person(pid)
            if p is None:
                continue
            jpg = self.memory.thumb(pid)
            out.append({"id": pid, "name": p.name, "code": p.code, "score": f"{score:.2f}",
                        "verdict": "MATCH" if score >= th.match else ("POSSIBLE" if score >= th.new else "UNLIKELY"),
                        "seen": p.sightings, "last": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.last_seen)),
                        "thumb": "data:image/jpeg;base64," + base64.b64encode(jpg).decode() if jpg else ""})
        return out

    def remember(self, key: str) -> str | None:
        from ..memory.linker import face_thumb
        MATCH = self.memory.th.match
        r = self.face(key)
        if r is None or self.memory is None:
            return None
        it, f = r
        j = it.faces.index(f)
        if j >= len(it.embs):
            return None
        e, yaw, q = it.embs[j], (f.pose.yaw if f.pose else 0.0), f.quality.get("overall", 0.5)
        thumb = face_thumb(it.image, f.box)
        try:
            ts = it.path.stat().st_mtime
        except OSError:
            import time
            ts = time.time()
        b = self._best(it, j)
        if b and b[1] >= MATCH:
            pid, score, verb = b[0], b[1], "added to"
            self.memory.learn(pid, e, yaw, q, face_thumb(it.image, f.box, 72))
        else:
            pid, score, verb = self.memory.create_person([(e, yaw, q, face_thumb(it.image, f.box, 72))], thumb, q,
                                                         ts), 1.0, "new profile"
        self.memory.add_sighting(pid, "image", it.name, ts, ts, 1, score, thumb)
        self.memory.update_thumb(pid, thumb, q)
        self._cache.pop(("detail", key), None)
        p = self.memory.person(pid)
        self.on_change()
        return f"D{j + 1:02d} {verb} {p.name if p else pid} ({score:.2f})"

    def _detail(self, key: str) -> dict:
        ck = ("detail", key, self.memory.version if self.memory is not None else 0)
        if ck in self._cache:
            return self._cache[ck]
        it, f = self.face(key)
        crop, M = square_crop(it.image, f.box, 300, context=1.35)
        if f.mesh is not None:
            pts = (M[:, :2] @ f.mesh[:, :2].T).T + M[:, 2]
            dim = (crop * 0.75).astype(np.uint8)
            draw_mesh_np(dim, pts)
            crop = dim
        else:
            for k in f.kps:
                c = M[:, :2] @ k + M[:, 2]
                cv2.circle(crop, tuple(np.int32(c)), 3, (255, 255, 255), 1, cv2.LINE_AA)
        rows = [("SOURCE", it.name), ("CONFIDENCE", f"{f.score:.2f}"), ("FACE SIZE", f"{int(f.box[2])}x{int(f.box[3])} PX"),
                ("VISIBILITY", "PARTIAL (EDGE)" if f.truncated else "FULL"),
                ("LANDMARKS", "478-POINT MESH" if f.mesh is not None else "5-POINT ONLY")]
        if f.pose:
            rows.append(("HEAD POSE", f"Y {f.pose.yaw:+.0f}° P {f.pose.pitch:+.0f}° R {f.pose.roll:+.0f}°"))
        if f.gaze:
            rows.append(("GAZE", f"Y {f.gaze.yaw:+.0f}° P {f.gaze.pitch:+.0f}°" + ("" if f.gaze.reliable else " (LOW CONF)")))
        if f.eye_open:
            rows.append(("EYE OPENNESS", f"R {f.eye_open[0]:.2f} · L {f.eye_open[1]:.2f}"))
        if f.blend:
            top = sorted(((k, v) for k, v in f.blend.items() if k != "_neutral"), key=lambda kv: -kv[1])[:3]
            rows.append(("EXPRESSION", " · ".join(f"{k} {v:.2f}" for k, v in top)))
        for k in ("overall", "sharpness", "exposure", "resolution", "frontality"):
            if k in f.quality:
                rows.append((f"Q.{k.upper()}", f"{f.quality[k]:.2f}"))
        mt = self._match_text(it, it.faces.index(f))
        if mt:
            rows.insert(0, ("MEMORY", mt))
        d = {"title": f"D{int(key.split(':')[1]) + 1:02d} · {it.name}", "img": jpg_data_url(crop), "rows": rows,
             "key": key, "can_remember": self.memory is not None, "matches": self._search(it, f)}
        self._cache[ck] = d
        return d

    def _compare(self, ka: str, kb: str) -> dict:
        (ia, fa), (ib, fb) = self.face(ka), self.face(kb)
        ca, _ = aligned_crop(ia.image, fa.kps)
        cb, _ = aligned_crop(ib.image, fb.kps)
        out = {"a_img": jpg_data_url(ca), "b_img": jpg_data_url(cb),
               "a_name": f"D{int(ka.split(':')[1]) + 1:02d} {ia.name}", "b_name": f"D{int(kb.split(':')[1]) + 1:02d} {ib.name}",
               "rows": [], "shape": ""}
        rows = out["rows"]

        def add(name, a, b, fmt="{:.2f}", rel=True, thr=0.1):
            if a is None or b is None:
                return
            d = b - a
            flag = abs(d) / max(abs(a), 1e-6) > thr if rel else abs(d) > thr
            rows.append([name, fmt.format(a), fmt.format(b), fmt.format(d), bool(flag)])

        if fa.mesh is not None and fb.mesh is not None:
            dist, b_al = geo.procrustes_distance(fa.mesh, fb.mesh)
            a_n = geo.normalized_shape(fa.mesh)
            canvas = np.full((260, 300, 3), 11, np.uint8)
            allp = np.vstack([a_n, b_al])
            lo, hi = allp.min(0), allp.max(0)
            sc = 230 / max(hi - lo)
            off = np.array([150, 130]) - (lo + hi) / 2 * sc
            for pts, col in ((a_n, (255, 255, 255)), (b_al, (40, 40, 255))):
                P = pts * sc + off
                for a, b in TESSELATION[::2]:
                    cv2.line(canvas, tuple(np.int32(P[a] * 4)), tuple(np.int32(P[b] * 4)), col, 1, cv2.LINE_AA, shift=2)
            cv2.putText(canvas, "A", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "B", (26, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1, cv2.LINE_AA)
            out["shape"] = png_data_url(canvas)
            rows.append(["SHAPE DIVERGENCE", "", "", f"{dist:.3f}", dist > 0.06])
            for k, label in (("face_width", "FACE WIDTH /IOD"), ("face_height", "FACE HEIGHT /IOD"), ("aspect", "ASPECT H/W"),
                             ("eye_width_r", "EYE WIDTH R"), ("eye_width_l", "EYE WIDTH L"), ("nose_width", "NOSE WIDTH"),
                             ("nose_length", "NOSE LENGTH"), ("mouth_width", "MOUTH WIDTH"), ("chin_mouth", "LIP-CHIN"),
                             ("brow_eye_r", "BROW-EYE R"), ("brow_eye_l", "BROW-EYE L"), ("mouth_open", "MOUTH OPEN")):
                add(label, fa.metrics.get(k), fb.metrics.get(k), thr=0.12)
        if fa.pose and fb.pose:
            add("YAW °", fa.pose.yaw, fb.pose.yaw, "{:+.0f}", rel=False, thr=15)
            add("PITCH °", fa.pose.pitch, fb.pose.pitch, "{:+.0f}", rel=False, thr=15)
            add("ROLL °", fa.pose.roll, fb.pose.roll, "{:+.0f}", rel=False, thr=15)
        if fa.eye_open and fb.eye_open:
            add("EYE OPEN (MEAN)", float(np.mean(fa.eye_open)), float(np.mean(fb.eye_open)), thr=0.25)
        for k in ("overall", "sharpness", "resolution", "exposure"):
            add(f"Q.{k.upper()}", fa.quality.get(k), fb.quality.get(k), rel=False, thr=0.2)
        add("FACE HEIGHT PX", float(fa.box[3]), float(fb.box[3]), "{:.0f}", thr=0.5)
        caveats = []
        if fa.pose and fb.pose and (abs(fa.pose.yaw - fb.pose.yaw) > 15 or abs(fa.pose.pitch - fb.pose.pitch) > 15):
            caveats.append("pose differs, so proportions are not directly comparable")
        if fa.mesh is None or fb.mesh is None:
            caveats.append("dense landmarks unavailable for one detection")
        out["note"] = ("OBSERVABLE-FEATURE COMPARISON ONLY. NOT AN IDENTITY MATCH. Proportions are normalised by "
                       "inter-ocular distance and are affected by pose, expression, lens and resolution."
                       + (" Caveat: " + "; ".join(caveats) + "." if caveats else ""))
        return out


class ImageView(QWidget):
    faceClicked = Signal(str, bool)

    def __init__(self, lab: ImageLab, layers: Layers, parent=None):
        super().__init__(parent)
        self.lab = lab
        self.layers = layers
        self._xf: Xf | None = None
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        p.fillRect(self.rect(), PAGE_BG)
        lab = self.lab
        if not lab.items:
            p.setFont(A.mono(14))
            p.setPen(A.BLACK)
            p.drawText(self.rect(), Qt.AlignCenter, "NO IMAGES\n\nLOAD PHOTOGRAPHS OR A FOLDER")
            p.end()
            return
        it = lab.items[min(lab.current, len(lab.items) - 1)]
        W, H = self.width(), self.height()
        if it.image is None:
            p.setFont(A.mono(12))
            p.setPen(A.BLACK)
            p.drawText(self.rect(), Qt.AlignCenter, "ERROR: " + it.error.upper() if it.error else "ANALYSING _")
            p.end()
            return
        fh, fw = it.image.shape[:2]
        s = min(W / fw, H / fh)
        ox, oy = (W - fw * s) / 2, (H - fh * s) / 2
        xf = self._xf = Xf(s, ox, oy)
        p.fillRect(QRectF(ox, oy, fw * s, fh * s), A.BLACK)
        draw_frame(p, it.image, QRectF(ox, oy, fw * s, fh * s))
        idx = lab.items.index(it)
        for j, f in enumerate(it.faces):
            key = f"{idx}:{j}"
            ab = next((k for k, v in lab.ab.items() if v == key), None)
            sub = [f"{f.score:.2f}"] + ([f"SET {ab}"] if ab else [])
            draw_face(p, xf, f, self.layers, label=f"D{j + 1:02d}", focused=key == lab.inspected or ab == "B",
                      sub=sub, blink=False)
        tag(p, 10, 10, f"{it.name.upper()}  ·  {len(it.faces)} FACE(S)", A.mono(9))
        p.end()

    def mousePressEvent(self, e):
        lab = self.lab
        if not lab.items or self._xf is None:
            return
        it = lab.items[min(lab.current, len(lab.items) - 1)]
        idx = lab.items.index(it)
        pos = e.position()
        best = None
        for j, f in enumerate(it.faces):
            r = self._xf.r(f.box).adjusted(-10, -10, 10, 10)
            if r.contains(pos):
                d = (r.center() - pos).manhattanLength()
                if best is None or d < best[0]:
                    best = (d, j)
        if best:
            self.faceClicked.emit(f"{idx}:{best[1]}", bool(e.modifiers() & Qt.ShiftModifier))
