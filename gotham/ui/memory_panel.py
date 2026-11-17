from __future__ import annotations

import base64
import threading
import time

import cv2
import numpy as np

from ..analysis import geometry as geo
from ..memory.store import YAW_BINS, MemoryStore
from .overlay import CONTOURS, TESSELATION

BIN_NAMES = ["L-PROFILE", "L-3/4", "FRONTAL", "R-3/4", "R-PROFILE"]
METRICS = [("face_width", "FACE WIDTH"), ("face_height", "FACE HEIGHT"), ("aspect", "ASPECT H/W"),
           ("eye_width_r", "EYE WIDTH R"), ("eye_width_l", "EYE WIDTH L"), ("nose_width", "NOSE WIDTH"),
           ("nose_length", "NOSE LENGTH"), ("mouth_width", "MOUTH WIDTH"), ("brow_eye_r", "BROW-EYE R"),
           ("brow_eye_l", "BROW-EYE L"), ("chin_mouth", "CHIN-MOUTH")]
REBUILD_EVERY = 1.0


def _url(jpeg: bytes | None) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode() if jpeg else ""


def _png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _ts(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def _dur(s: float) -> str:
    s = max(0, int(round(s)))
    if s >= 86400:
        return f"{s // 86400}d{s % 86400 // 3600:02d}h"
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _ago(t: float) -> str:
    d = time.time() - t
    if d < 90:
        return "JUST NOW"
    if d < 5400:
        return f"{d / 60:.0f} MIN AGO"
    if d < 129600:
        return f"{d / 3600:.0f} H AGO"
    return f"{d / 86400:.0f} DAYS AGO"


def render_wireframe(mesh: np.ndarray, w: int = 320, h: int = 360) -> np.ndarray:
    P = mesh[:, :2].astype(np.float64)
    lo, hi = P.min(0), P.max(0)
    s = min((w - 36) / max(hi[0] - lo[0], 1e-6), (h - 36) / max(hi[1] - lo[1], 1e-6))
    X = (P - (lo + hi) / 2) * s + (w / 2, h / 2)
    img = np.zeros((h, w, 3), np.uint8)
    for g in range(0, max(w, h), 24):
        cv2.line(img, (g, 0), (g, h), (22, 22, 22), 1)
        cv2.line(img, (0, g), (w, g), (22, 22, 22), 1)
    pts = np.round(X * 16).astype(np.int32)
    z = mesh[:, 2]
    zn = (z - z.min()) / max(float(np.ptp(z)), 1e-6)
    ez = (zn[TESSELATION[:, 0]] + zn[TESSELATION[:, 1]]) / 2
    for k in range(6):
        sel = TESSELATION[(ez >= k / 6) & (ez < (k + 1) / 6 + (k == 5))]
        if len(sel):
            v = int(205 - k * 30)
            cv2.polylines(img, list(np.stack([pts[sel[:, 0]], pts[sel[:, 1]]], 1)), False, (v, v, v), 1,
                          cv2.LINE_AA, shift=4)
    cv2.polylines(img, list(np.stack([pts[CONTOURS[:, 0]], pts[CONTOURS[:, 1]]], 1)), False, (255, 255, 255), 1,
                  cv2.LINE_AA, shift=4)
    for e in (geo.R_EYE, geo.L_EYE):
        if mesh.shape[0] > e["iris"]:
            c = X[e["iris"]]
            r = float(np.mean([np.linalg.norm(X[i] - c) for i in e["iris_ring"]]))
            cv2.circle(img, (int(c[0] * 16), int(c[1] * 16)), max(16, int(r * 16)), (40, 40, 255), 1, cv2.LINE_AA, 4)
    for i in (geo.NOSE_TIP, geo.CHIN, geo.FOREHEAD, geo.CHEEK_R, geo.CHEEK_L):
        c = X[i]
        cv2.rectangle(img, (int(c[0]) - 2, int(c[1]) - 2), (int(c[0]) + 2, int(c[1]) + 2), (40, 40, 255), -1)
    return img


def render_signature(v: np.ndarray, h: int = 18) -> np.ndarray:
    a = v / (np.abs(v).max() + 1e-9)
    img = np.zeros((h, len(a), 3), np.uint8)
    pos, neg = np.clip(a, 0, 1), np.clip(-a, 0, 1)
    img[:, :, 0] = img[:, :, 1] = (pos * 255).astype(np.uint8)[None]
    img[:, :, 2] = (np.maximum(pos, neg * 0.9) * 255).astype(np.uint8)[None]
    return img


class MemoryPanel:
    def __init__(self, store: MemoryStore | None):
        self.store = store
        self.selected: int | None = None
        self.page: int | None = None
        self.review = False
        self._ui_key = None
        self._data_key = None
        self._built = 0.0
        self._payload: dict = {}
        self._thumbs: dict[int, tuple[float, str]] = {}
        self._detail_key = None
        self._detail: dict | None = None
        self._page_key = None
        self._page_built = 0.0
        self._page: dict | None = None
        self._sthumbs: dict[int, tuple[int, str]] = {}
        self._sent: dict[int, float] = {}
        self._dups: list = []
        self._dup_key = None
        self._dup_gen = 0
        self._dup_t = 0.0
        self._dup_busy = False

    def page_reloaded(self):
        self._sent.clear()
        self._ui_key = None

    def summary(self) -> dict:
        if self.store is None:
            return {"available": False}
        st = self.store.stats()
        return {"available": True, "enabled": self.store.enabled, "profiles": st["profiles"],
                "upgrading": None if self.store.migration_progress is None else round(self.store.migration_progress * 100),
                "sightings": st["sightings"]}

    def payload(self, live: dict[int, list[str]]) -> dict:
        s = self.store
        if s is None:
            return {"available": False}
        ui_key = (self.selected, self.page, self.review)
        self._refresh_dups(urgent=ui_key != self._ui_key)
        data_key = (s.version, self._dup_gen, tuple(sorted((k, tuple(v)) for k, v in live.items())))
        now = time.monotonic()
        every = REBUILD_EVERY * (2 if len(self._payload.get("people", ())) > 150 else 1)
        if ui_key == self._ui_key and (data_key == self._data_key or now - self._built < every):
            return self._payload
        people = []
        for p in s.persons():
            ver = p.thumb_q
            if self._thumbs.get(p.id, (None,))[0] != ver:
                self._thumbs[p.id] = (ver, _url(s.thumb(p.id)))
            people.append({"id": p.id, "code": p.code, "name": p.name, "label": p.label, "thumb": self._thumbs[p.id][1],
                           "sightings": p.sightings, "first": _ts(p.first_seen), "last": _ts(p.last_seen),
                           "fs": p.first_seen, "ls": p.last_seen,
                           "ago": _ago(p.last_seen), "live": live.get(p.id), "alert": p.alert,
                           "bins": [i in p.bins for i in range(len(YAW_BINS) - 1)]})
        by_id = {x["id"]: x for x in people}
        gallery = []
        for x in people:
            ver = self._thumbs[x["id"]][0]
            if self._sent.get(x["id"]) == ver:
                gallery.append({**x, "thumb": None})
            else:
                gallery.append(x)
                self._sent[x["id"]] = ver
        if self.selected is not None and self.selected not in by_id:
            self.selected = None
        if self.page is not None and self.page not in by_id:
            self.page = None
        detail = None
        if self.selected is not None and self.page is None:
            dkey = (s.version, self.selected)
            if dkey != self._detail_key:
                p = by_id[self.selected]
                hist = s.sightings(self.selected, 200)
                views = [{"thumb": _url(v["thumb"]), "yaw": f"{v['yaw']:+.0f}"} for v in s.views(self.selected)
                         if v["thumb"]]
                self._detail = {
                    **p, "bin_names": BIN_NAMES, "views": views, "similar": self._similar(self.selected, by_id, 3),
                    "model": s.th.display,
                    "history": [{"thumb": _url(h["thumb"]), "source": h["source"], "kind": h["kind"],
                                 "from": _ts(h["start"]), "to": time.strftime("%H:%M:%S", time.localtime(h["end"])),
                                 "dur": _dur(h["end"] - h["start"]), "score": f"{h['score'] or 0:.2f}"}
                                for h in hist]}
                self._detail_key = dkey
            detail = {**self._detail, "live": live.get(self.selected)}
        page = None
        if self.page is not None:
            if (self.page, s.version) != self._page_key and \
                    (self._page_key is None or self._page_key[0] != self.page or now - self._page_built >= REBUILD_EVERY):
                self._page = self._build_page(self.page, by_id)
                self._page_key, self._page_built = (self.page, s.version), now
            if self._page is not None:
                page = {**self._page, "live": live.get(self.page)}
        all_dups = [d for d in self._dups if d[0] in by_id and d[1] in by_id]
        dups = all_dups[:20]
        certain = sum(1 for d in all_dups if s.is_certain(d[2], d[3]))
        thumbs = {x["id"]: x["thumb"] for x in people}
        names = {x["id"]: x["name"] for x in people}
        review = [{"a": a, "b": b, "score": f"{sc:.2f}", "whole": f"{whole:.2f}", "certain": s.is_certain(sc, whole),
                   "a_name": names.get(a, a), "b_name": names.get(b, b),
                   "a_thumb": thumbs.get(a, ""), "b_thumb": thumbs.get(b, "")} for a, b, sc, whole in dups] if self.review else []
        recent = [{"id": x["id"], "code": x["code"], "name": x["name"], "thumb": x["thumb"], "ago": x["ago"],
                   "live": x["live"]} for x in people[:14]] if page else []
        self._payload = {"available": True, "enabled": s.enabled, "stats": s.stats(), "people": gallery,
                         "selected": self.selected, "detail": detail, "page": page, "recent": recent,
                         "dup_count": len(all_dups), "review": review, "reviewing": self.review,
                         "certain": certain,
                         "prunable": len(s.prunable(7))}
        self._ui_key, self._data_key, self._built = ui_key, data_key, now
        return self._payload

    def _refresh_dups(self, urgent: bool = False):
        s = self.store
        key = s.emb_version
        if key == self._dup_key or self._dup_busy or (not urgent and time.monotonic() - self._dup_t < 10):
            return
        self._dup_busy, self._dup_t = True, time.monotonic()

        def work():
            try:
                self._dups = s.duplicate_review()
                self._dup_key = key
            finally:
                self._dup_gen += 1
                self._dup_busy = False
        threading.Thread(target=work, daemon=True, name="duplicate-scan").start()

    def _similar(self, pid: int, by_id: dict, k: int) -> list[dict]:
        out = []
        for oid, score in self.store.similar(pid, k):
            o = by_id.get(oid)
            if o:
                out.append({"id": oid, "code": o["code"], "name": o["name"], "thumb": o["thumb"],
                            "score": f"{score:.2f}", "likely": score >= self.store.th.match})
        return out

    def _sighting_thumb(self, row: dict) -> str:
        c = self._sthumbs.get(row["id"])
        if c is None or c[0] != row["tsize"]:
            c = self._sthumbs[row["id"]] = (row["tsize"], _url(self.store.sighting_thumb(row["id"])) if row["tsize"] else "")
        return c[1]

    def _build_page(self, pid: int, by_id: dict) -> dict | None:
        s = self.store
        p = s.person(pid)
        if p is None:
            return None
        card = by_id.get(pid, {})
        st = s.profile_stats(pid)
        th = s.th
        face = {"wire": "", "metrics": [], "signature": "", "bins": st["bins"], "bin_names": BIN_NAMES,
                "coherence": None if st["coherence"] is None else round(st["coherence"], 2)}
        mesh = s.mesh(pid)
        if mesh is not None and len(mesh) >= 468:
            face["wire"] = _png(render_wireframe(mesh))
            m = geo.geometry_metrics(mesh)
            face["metrics"] = [[label, f"{m[k]:.2f}"] for k, label in METRICS if k in m]
        sig = s.signature(pid)
        if sig is not None:
            face["signature"] = _png(render_signature(sig))
        rows = s.sighting_rows(pid, 400)
        timeline = [{"id": r["id"], "thumb": self._sighting_thumb(r), "source": r["source"] or "?", "kind": r["kind"],
                     "from": _ts(r["start"]), "to": time.strftime("%H:%M:%S", time.localtime(r["end"])),
                     "dur": _dur(r["end"] - r["start"]), "score": f"{r['score']:.2f}", "s": round(r["score"], 4),
                     "t": r["start"], "frames": r["frames"] or 0} for r in rows]
        best = sorted(timeline, key=lambda x: -x["s"])[:6]
        cands = [{"id": c["id"], "thumb": _url(c["thumb"]), "source": c["source"] or "?", "when": _ts(c["t"]),
                  "ago": _ago(c["t"]), "score": f"{c['score']:.2f}", "label": c["label"] or "",
                  "other": by_id.get(c["other_id"], {}).get("name") if c["other_id"] else None,
                  "other_id": c["other_id"], "strength": max(0.0, min(1.0, (c["score"] - (th.match - 0.07)) / 0.07))}
                 for c in s.candidates(pid, 40)]
        today = time.time()
        days = [{"n": n, "d": time.strftime("%d", time.localtime(today - (13 - i) * 86400)),
                 "label": time.strftime("%a %d %b", time.localtime(today - (13 - i) * 86400))}
                for i, n in enumerate(st["days"])]
        hmax = max(st["hours"]) or 1.0
        stats = [["SIGHTINGS", str(st["sightings"])], ["TIME ON CAMERA", _dur(st["total_seconds"])],
                 ["SOURCES", str(len(st["sources"]))], ["FIRST SEEN", _ts(p.first_seen)],
                 ["LAST SEEN", f"{_ts(p.last_seen)} · {_ago(p.last_seen)}"], ["PROFILE CREATED", _ts(p.created)],
                 ["BEST MATCH", "-" if st["best_score"] is None else f"{st['best_score']:.2f}"],
                 ["AVG MATCH", "-" if st["mean_score"] is None else f"{st['mean_score']:.2f}"],
                 ["VIEWS STORED", f"{st['views']} · {sum(1 for b in st['bins'] if b)} OF {len(BIN_NAMES)} ANGLES"],
                 ["POSSIBLE SIGHTINGS", str(st["candidates"])]]
        report = {v["id"]: v for v in s.view_report(pid)}
        views = [{"id": v["id"], "thumb": _url(v["thumb"]), "yaw": f"{v['yaw']:+.0f}", "q": f"{v['quality']:.2f}",
                  "agree": None if report.get(v["id"], {}).get("agreement") is None else f"{report[v['id']]['agreement']:.2f}",
                  "stray": report.get(v["id"], {}).get("stray", False)}
                 for v in s.views(pid) if v["thumb"]]
        face["stray"] = sum(1 for v in report.values() if v["stray"])
        face["pair_disagrees"] = (len(report) == 2 and all(v["agreement"] is not None and v["agreement"] < s.STRAY_BELOW
                                                           for v in report.values()))
        return {
            "id": pid, "code": p.code, "name": p.name, "label": p.label, "notes": p.notes, "alert": p.alert,
            "thumb": card.get("thumb") or _url(s.thumb(pid)), "stats": stats, "face": face, "views": views,
            "best": best, "candidates": cands, "timeline": timeline, "similar": self._similar(pid, by_id, 5),
            "associates": [{"id": a["id"], "code": by_id[a["id"]]["code"], "name": by_id[a["id"]]["name"],
                            "thumb": by_id[a["id"]]["thumb"], "visits": a["visits"], "time": _dur(a["seconds"])}
                           for a in s.associates(pid, 6) if a["id"] in by_id],
            "sources": [{"source": x["source"], "n": x["n"], "time": _dur(x["seconds"]), "last": _ago(x["last"])}
                        for x in st["sources"]],
            "hours": [round(h / hmax, 3) for h in st["hours"]], "days": days, "model": th.display,
            "thresholds": {"match": th.match, "possible": round(th.match - 0.07, 2)},
        }
