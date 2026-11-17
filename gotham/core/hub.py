from __future__ import annotations

import itertools
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .feed import Feed, FeedEvent, TrackView

WATCH_KINDS = {
    "count_at_least": "subject count on the feed rises to at least `value`",
    "count_at_most": "subject count on the feed falls to at most `value`",
    "subject_acquired": "a new subject is acquired",
    "subject_lost": "a subject (or `subject`) leaves view",
    "subject_reacquired": "a departed subject (or `subject`) is re-acquired",
    "dwell_exceeds": "a subject (or `subject`) stays continuously visible longer than `value` seconds",
    "gaze_at_camera": "a subject (or `subject`) looks at the camera for over 1 second",
    "signal_lost": "a feed loses its signal",
    "person_seen": "a remembered person (`person`, e.g. P-0003 or their label) is recognised; or any known person",
    "new_person": "someone not in face memory is seen and a new profile is created",
    "person_moves": "a remembered person (or `person`) turns up on a different feed than where last seen",
}


@dataclass
class Watch:
    id: int
    kind: str
    feed_id: int | None = None
    subject: int | None = None
    value: float | None = None
    note: str = ""
    once: bool = False
    person: int | None = None
    created: float = field(default_factory=time.time)
    fired: int = 0
    last_fired: float | None = None
    active: bool = True
    _state: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {"watch_id": self.id, "kind": self.kind, "feed": _fname(self.feed_id), "subject": _sname(self.subject),
                "person": None if self.person is None else f"P-{self.person:04d}",
                "value": self.value, "note": self.note, "once": self.once, "times_fired": self.fired,
                "active": self.active}


@dataclass
class Alert:
    id: int
    wall: float
    watch_id: int
    feed_id: int | None
    subject: int | None
    title: str
    text: str


def _fname(fid):
    return None if fid is None else f"FEED-{fid:02d}"


def _sname(sid):
    return None if sid is None else f"S{sid:02d}"


def _round(v, n=1):
    return None if v is None else round(float(v), n)


class Hub:
    def __init__(self):
        self.feeds: dict[int, Feed] = {}
        self._lock = threading.RLock()
        self.events: deque[FeedEvent] = deque(maxlen=5000)
        self._event_seq = itertools.count(1)
        self.watches: dict[int, Watch] = {}
        self._watch_ids = itertools.count(1)
        self._alert_ids = itertools.count(1)
        self._alerts: deque[Alert] = deque(maxlen=500)
        self._pending_alerts: deque[Alert] = deque()
        self._pending_events: deque[FeedEvent] = deque()
        self.focus: tuple[int, int] | None = None
        self.started = time.time()
        self._last_loc: dict[int, tuple[int, float]] = {}
        self._watchlist_last: dict[int, float] = {}
        self.activity: dict[int, deque] = {}
        self._activity_t = 0.0
        self.ui_actions: deque[dict] = deque()
        self.memory = None
        self.display = None
        self.display_help: dict[str, str] = {}

    def add_feed(self, feed: Feed):
        feed.on_event = self._on_event
        feed.memory = self.memory
        with self._lock:
            self.feeds[feed.id] = feed

    def remove_feed(self, feed_id: int):
        with self._lock:
            f = self.feeds.pop(feed_id, None)
        if f:
            f.stop()

    def feed(self, ref) -> Feed | None:
        fid = self._parse_feed(ref)
        return self.feeds.get(fid) if fid is not None else None

    @staticmethod
    def _parse_feed(ref) -> int | None:
        if ref is None:
            return None
        if isinstance(ref, int):
            return ref
        s = str(ref).upper().replace("FEED", "").replace("-", "").replace("_", "").strip()
        return int(s) if s.isdigit() else None

    @staticmethod
    def _parse_subject(ref) -> int | None:
        if ref is None:
            return None
        if isinstance(ref, int):
            return ref
        s = str(ref).upper().split("/")[-1].replace("S", "").replace("-", "").strip()
        return int(s) if s.isdigit() else None

    def _on_event(self, feed: Feed, ev: FeedEvent):
        with self._lock:
            self.events.append(ev)
            self._pending_events.append(ev)
            self._watch_event(ev)
            if ev.kind == "recognized" and ev.person and self.memory is not None \
                    and ev.person in self.memory.alert_ids() and ev.wall - self._watchlist_last.get(ev.person, 0) > 30:
                self._watchlist_last[ev.person] = ev.wall
                a = Alert(next(self._alert_ids), time.time(), 0, ev.feed_id, ev.subject, "WATCHLIST",
                          f"{_fname(ev.feed_id)} {self.person_name(ev.person)} sighted".strip())
                self._alerts.append(a)
                self._pending_alerts.append(a)
            if ev.kind in ("recognized", "new_person") and ev.person:
                last = self._last_loc.get(ev.person)
                if last and last[0] != ev.feed_id and 1.0 < ev.wall - last[1] < 1800:
                    dt = ev.wall - last[1]
                    name = self.person_name(ev.person)
                    tr = FeedEvent(ev.t, ev.wall, ev.feed_id, "transit", ev.subject,
                                   f"{name} moved {_fname(last[0])} -> {_fname(ev.feed_id)} in {dt:.0f}s", ev.person)
                    self.events.append(tr)
                    self._pending_events.append(tr)
                    self._watch_event(tr)
                self._last_loc[ev.person] = (ev.feed_id, ev.wall)

    def _watch_event(self, ev: FeedEvent):
        for w in list(self.watches.values()):
            if not w.active or (w.feed_id is not None and w.feed_id != ev.feed_id):
                continue
            if w.subject is not None and ev.subject != w.subject:
                continue
            if w.kind == "person_seen":
                if ev.kind == "recognized" and (w.person is None or w.person == ev.person):
                    self._fire(w, ev.feed_id, ev.subject, ev.text)
                continue
            if w.kind == "new_person":
                if ev.kind == "new_person":
                    self._fire(w, ev.feed_id, ev.subject, ev.text)
                continue
            if w.kind == "person_moves":
                if ev.kind == "transit" and (w.person is None or w.person == ev.person):
                    self._fire(w, ev.feed_id, ev.subject, ev.text)
                continue
            if (w.kind == "subject_acquired" and ev.kind == "acquired") or \
               (w.kind == "subject_lost" and ev.kind == "lost") or \
               (w.kind == "subject_reacquired" and ev.kind == "reacquired") or \
               (w.kind == "signal_lost" and ev.kind == "signal_lost"):
                self._fire(w, ev.feed_id, ev.subject, ev.text)

    def _fire(self, w: Watch, feed_id, subject, text):
        w.fired += 1
        w.last_fired = time.time()
        title = w.kind.replace("_", " ").upper()
        a = Alert(next(self._alert_ids), time.time(), w.id, feed_id, subject, title,
                  f"{_fname(feed_id) or ''} {text}".strip() + (f" — {w.note}" if w.note else ""))
        self._alerts.append(a)
        self._pending_alerts.append(a)
        if w.once:
            w.active = False

    def drain(self) -> tuple[list[FeedEvent], list[Alert]]:
        with self._lock:
            ev, al = list(self._pending_events), list(self._pending_alerts)
            self._pending_events.clear()
            self._pending_alerts.clear()
        return ev, al

    def tick(self):
        now = time.monotonic()
        stamp = time.time() - self._activity_t >= 1.0
        if stamp:
            self._activity_t = time.time()
        with self._lock:
            watches = [w for w in self.watches.values() if w.active]
            feeds = list(self.feeds.values())
        for f in feeds:
            r = f.latest()
            if r is None or not f.tracking:
                continue
            visible = [t for t in r.tracks if t.visible and not t.suppressed]
            n = len(visible)
            age = now - r.t
            fresh = age < 1.0
            frame_wall = time.time() - age
            if fresh:
                for t in visible:
                    if t.person_id is not None:
                        self._last_loc[t.person_id] = (f.id, frame_wall)
            if stamp or f.id not in self.activity:
                self.activity.setdefault(f.id, deque(maxlen=3600)).append((time.time(), n if fresh else 0))
            for w in watches:
                if w.feed_id is not None and w.feed_id != f.id:
                    continue
                key = f.id
                if w.kind in ("count_at_least", "count_at_most"):
                    cond = n >= (w.value or 0) if w.kind == "count_at_least" else n <= (w.value or 0)
                    was = w._state.get(key, False)
                    if cond and was is False:
                        self._fire(w, f.id, None, f"{n} subject(s) in view")
                    w._state[key] = cond
                elif w.kind == "dwell_exceeds":
                    fired = w._state.setdefault(("fired", key), set())
                    for t in visible:
                        if w.subject is not None and t.id != w.subject:
                            continue
                        start = self._visible_since(t)
                        if now - start >= (w.value or 30) and t.id not in fired:
                            fired.add(t.id)
                            self._fire(w, f.id, t.id, f"{t.label} in view for {now - start:.0f}s")
                elif w.kind == "gaze_at_camera":
                    since = w._state.setdefault(("since", key), {})
                    fired = w._state.setdefault(("fired", key), set())
                    for t in visible:
                        if w.subject is not None and t.id != w.subject:
                            continue
                        g = t.face.gaze
                        at = g is not None and g.reliable and abs(g.yaw) < 12 and abs(g.pitch) < 12
                        if at:
                            since.setdefault(t.id, now)
                            if now - since[t.id] > 1.0 and t.id not in fired:
                                fired.add(t.id)
                                self._fire(w, f.id, t.id, f"{t.label} looking at camera")
                        else:
                            since.pop(t.id, None)
                            fired.discard(t.id)

    @staticmethod
    def _visible_since(t: TrackView) -> float:
        trail = t.trail
        if not trail:
            return t.last_seen
        start = trail[-1][0]
        for i in range(len(trail) - 1, 0, -1):
            if trail[i][0] - trail[i - 1][0] > 1.5:
                break
            start = trail[i - 1][0]
        return start

    def overview(self) -> dict:
        with self._lock:
            feeds = list(self.feeds.values())
        out = []
        for f in feeds:
            r = f.latest()
            vis = [t for t in (r.tracks if r else []) if t.visible and not t.suppressed]
            out.append({
                "feed": f.name, "source": f.spec.label, "kind": f.spec.kind.value,
                "status": f.stats.status.value, "tracking_session": f.tracking,
                "faces_detected_now": len(r.faces) if r else 0,
                "subjects_visible_now": len(vis),
                "subjects_in_session": f.tracker.total_subjects if f.tracking else 0,
            })
        return {"system_uptime_s": round(time.time() - self.started), "feeds": out,
                "active_watches": sum(1 for w in self.watches.values() if w.active),
                "focus": None if not self.focus else f"{_fname(self.focus[0])}/{_sname(self.focus[1])}"}

    def feed_status(self, ref=None) -> list[dict]:
        out = []
        for f in self._feeds(ref):
            s = f.stats
            r = f.latest()
            up = None if s.connected_at is None else round(time.monotonic() - s.connected_at)
            out.append({"feed": f.name, "source": f.spec.label, "kind": f.spec.kind.value, "status": s.status.value,
                        "resolution": f"{s.width}x{s.height}", "capture_fps": _round(s.fps),
                        "analysis_fps": _round(r.process_fps if r else 0), "latency_ms": _round(r.latency_ms if r else None, 0),
                        "frames": s.frames, "dropped_frames": s.dropped, "reconnects": s.reconnects,
                        "connected_for_s": up, "error": s.error or None, "inference": f.provider,
                        "small_face_mode": f.settings.small_faces, "landmarks": f.settings.landmarks})
        return out

    def _feeds(self, ref) -> list[Feed]:
        with self._lock:
            if ref is None:
                return list(self.feeds.values())
            f = self.feed(ref)
            return [f] if f else []

    _names: dict = {}
    _names_ver = -1

    @staticmethod
    def _person_fields(t: TrackView) -> dict:
        if t.person_id is None:
            return {"memory": t.memory_state} if t.memory_state else {}
        n = Hub._names.get(t.person_id)
        return {"person": n or f"P-{t.person_id:04d}", "profile": f"P-{t.person_id:04d}",
                "match_score": round(float(t.person_score), 2), "memory": t.memory_state}

    _info: dict = {}
    _names_t = 0.0

    def refresh_names(self):
        now = time.monotonic()
        if self.memory is not None and self.memory.version != Hub._names_ver and \
                (now - Hub._names_t > 0.5 or not Hub._names):
            ps = self.memory.persons()
            Hub._names = {p.id: p.name for p in ps}
            Hub._info = {p.id: {"name": p.name, "label": p.label, "code": p.code, "sightings": p.sightings,
                                "first_seen": p.first_seen, "last_seen": p.last_seen, "views": p.exemplars} for p in ps}
            Hub._names_ver = self.memory.version
            Hub._names_t = now

    def person_info(self, pid) -> dict | None:
        if pid is None:
            return None
        self.refresh_names()
        return Hub._info.get(pid)

    def activity_summary(self, ref=None, minutes: float = 10) -> list[dict]:
        out = []
        since = time.time() - minutes * 60
        for f in self._feeds(ref):
            pts = [(t, n) for t, n in self.activity.get(f.id, ()) if t >= since]
            if not pts:
                out.append({"feed": f.name, "samples": 0})
                continue
            buckets: dict[int, list[int]] = {}
            for t, n in pts:
                buckets.setdefault(int((t - since) // 60), []).append(n)
            out.append({"feed": f.name, "now": pts[-1][1], "peak": max(n for _, n in pts),
                        "average": round(sum(n for _, n in pts) / len(pts), 2),
                        "per_minute_peak": [max(buckets.get(i, [0])) for i in range(int(minutes))],
                        "tracking": f.tracking})
        return out

    def person_name(self, pid):
        if pid is None:
            return None
        self.refresh_names()
        return Hub._names.get(pid, f"P-{pid:04d}")

    def parse_person(self, ref):
        if ref is None or self.memory is None:
            return None
        if isinstance(ref, int):
            return ref
        r = str(ref).strip()
        m = re.fullmatch(r"(?i)p-?0*(\d+)", r)
        if m:
            return int(m.group(1))
        for p in self.memory.persons():
            if p.label and p.label.lower() == r.lower():
                return p.id
        return None

    @staticmethod
    def _ts(t: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

    def _person_dict(self, p, live: dict) -> dict:
        return {"profile": p.code, "label": p.label or None, "first_seen": self._ts(p.first_seen),
                "last_seen": self._ts(p.last_seen), "sightings": p.sightings, "views_stored": p.exemplars,
                "in_view_now": live.get(p.id)}

    def _live_people(self) -> dict:
        live = {}
        for f in list(self.feeds.values()):
            r = f.latest()
            for t in (r.tracks if r else []):
                if t.person_id is not None and t.visible:
                    live.setdefault(t.person_id, []).append(f"{f.name}/{t.label}")
        return live

    def people(self, query=None, limit: int = 50) -> dict:
        if self.memory is None:
            return {"error": "face memory is off"}
        live = self._live_people()
        allp = self.memory.persons()
        ps = allp
        if query:
            q = query.lower()
            ps = [p for p in allp if q in p.code.lower() or q in (p.label or "").lower()]
        return {"total_profiles": len(allp), "recognition_enabled": self.memory.enabled,
                "people": [self._person_dict(p, live) for p in ps[:limit]]}

    def person_detail(self, ref) -> dict:
        pid = self.parse_person(ref)
        p = self.memory.person(pid) if pid is not None and self.memory else None
        if p is None:
            return {"error": f"no profile {ref!r}"}
        d = self._person_dict(p, self._live_people())
        st = self.memory.profile_stats(pid)
        d.update({
            "operator_notes": p.notes or None, "watchlist": p.alert,
            "time_on_camera_seconds": round(st["total_seconds"]),
            "best_match_score": round(st["best_score"], 2) if st["best_score"] is not None else None,
            "sources": [{"source": s["source"], "sightings": s["n"], "seconds": round(s["seconds"]),
                         "last": self._ts(s["last"])} for s in st["sources"][:8]],
            "busiest_hour": int(max(range(24), key=lambda h: st["hours"][h])) if st["sightings"] else None,
            "profile_consistency": round(st["coherence"], 2) if st["coherence"] is not None else None,
            "possible_sightings": [{"source": c["source"], "time": self._ts(c["t"]), "similarity": round(c["score"], 2),
                                    "subject": c["label"], "became": f"P-{c['other_id']:04d}" if c["other_id"] else None}
                                   for c in self.memory.candidates(pid, 10)],
        })
        d["history"] = [{"source": h["source"], "kind": h["kind"], "from": self._ts(h["start"]),
                         "to": self._ts(h["end"]), "seconds": round(h["end"] - h["start"]),
                         "match_score": round(h["score"] or 0, 2)} for h in self.memory.sighting_rows(pid, 30)]
        return d

    def people_seen(self, seconds: float = 3600) -> dict:
        if self.memory is None:
            return {"error": "face memory is off"}
        since = time.time() - seconds
        out: dict = {}
        for h in self.memory.recent_sightings(since):
            e = out.setdefault(h["person_id"], {"person": self.person_name(h["person_id"]), "sightings": 0,
                                                "sources": set(), "last": 0})
            e["sightings"] += 1
            e["sources"].add(h["source"])
            e["last"] = max(e["last"], h["end"])
        for pid, where in self._live_people().items():
            e = out.setdefault(pid, {"person": self.person_name(pid), "sightings": 0, "sources": set(),
                                     "last": time.time()})
            e["in_view_now"] = where
        return {"window_seconds": seconds,
                "people": [{**v, "sources": sorted(v["sources"]), "last": self._ts(v["last"])}
                           for v in sorted(out.values(), key=lambda v: -v["last"])]}

    def label_person(self, ref, label: str) -> dict:
        pid = self.parse_person(ref)
        if pid is None or self.memory is None or self.memory.person(pid) is None:
            return {"error": f"no profile {ref!r}"}
        self.memory.rename(pid, label)
        return {"profile": f"P-{pid:04d}", "label": label}

    @staticmethod
    def describe_track(t: TrackView, now: float, detail: bool = False) -> dict:
        f = t.face
        d = {"subject": t.label, "state": t.state, "visible": t.visible, **Hub._person_fields(t),
             "seconds_in_session": round(now - t.first_seen, 1),
             "seconds_since_seen": round(max(0.0, now - t.last_seen), 1),
             "reacquisitions": t.reacquired}
        x, y, w, h = (float(v) for v in t.box)
        d["position_px"] = [round(x + w / 2), round(y + h / 2)]
        d["face_height_px"] = round(h)
        spd = float(np.hypot(*t.velocity))
        d["speed_px_s"] = round(spd)
        if spd > 15:
            ang = np.degrees(np.arctan2(t.velocity[1], t.velocity[0]))
            dirs = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
            d["moving"] = dirs[int(((ang + 22.5) % 360) // 45)]
        if f.pose:
            d["head_pose_deg"] = {"yaw": _round(f.pose.yaw), "pitch": _round(f.pose.pitch), "roll": _round(f.pose.roll)}
        if f.gaze:
            d["gaze_deg"] = {"yaw": _round(f.gaze.yaw), "pitch": _round(f.gaze.pitch), "reliable": f.gaze.reliable}
            d["looking_at_camera"] = bool(f.gaze.reliable and abs(f.gaze.yaw) < 12 and abs(f.gaze.pitch) < 12)
        if f.quality:
            d["quality"] = round(f.quality.get("overall", 0), 2)
        if detail:
            d["quality_breakdown"] = {k: round(v, 2) for k, v in f.quality.items()}
            d["detection_confidence"] = round(f.score, 2)
            d["partially_out_of_frame"] = f.truncated
            if f.eye_open:
                d["eye_openness"] = [round(v, 2) for v in f.eye_open]
            if f.blend:
                top = sorted(((k, v) for k, v in f.blend.items() if k != "_neutral"), key=lambda kv: -kv[1])[:5]
                d["expression_blendshapes_top"] = {k: round(v, 2) for k, v in top}
            d["dense_landmarks"] = f.mesh is not None
        return d

    def subjects(self, ref=None, include_departed=False) -> list[dict]:
        self.refresh_names()
        now = time.monotonic()
        out = []
        for f in self._feeds(ref):
            r = f.latest()
            if r is None:
                continue
            for t in r.tracks:
                if t.suppressed or (t.state == "lost" and not include_departed):
                    continue
                d = self.describe_track(t, now)
                d["feed"] = f.name
                out.append(d)
        return out

    def subject(self, feed_ref, subject_ref) -> dict | None:
        self.refresh_names()
        sid = self._parse_subject(subject_ref)
        feeds = self._feeds(feed_ref)
        for f in feeds:
            r = f.latest()
            for t in (r.tracks if r else []):
                if t.id == sid and not t.suppressed:
                    d = self.describe_track(t, time.monotonic(), detail=True)
                    d["feed"] = f.name
                    d["recent_events"] = [self._ev(e) for e in list(f.events)[-200:] if e.subject == sid][-10:]
                    return d
        return None

    def faces_now(self, ref=None) -> list[dict]:
        out = []
        for f in self._feeds(ref):
            r = f.latest()
            for i, face in enumerate(r.faces if r else []):
                d = {"feed": f.name, "index": i, "confidence": round(face.score, 2),
                     "face_height_px": round(float(face.box[3])), "quality": round(face.quality.get("overall", 0), 2),
                     "partially_out_of_frame": face.truncated}
                if face.pose:
                    d["head_yaw_deg"] = _round(face.pose.yaw)
                if face.gaze:
                    d["looking_at_camera"] = bool(face.gaze.reliable and abs(face.gaze.yaw) < 12 and abs(face.gaze.pitch) < 12)
                out.append(d)
        return out

    @staticmethod
    def _ev(e: FeedEvent) -> dict:
        return {"time": time.strftime("%H:%M:%S", time.localtime(e.wall)), "feed": _fname(e.feed_id),
                "event": e.kind, "subject": _sname(e.subject), "detail": e.text,
                **({"person": f"P-{e.person:04d}"} if e.person else {})}

    def recent_events(self, seconds: float = 300, ref=None, kinds=None, limit: int = 60) -> list[dict]:
        cutoff = time.time() - seconds
        fid = self._parse_feed(ref)
        with self._lock:
            evs = [e for e in self.events if e.wall >= cutoff and (fid is None or e.feed_id == fid)
                   and (not kinds or e.kind in kinds)]
        return [self._ev(e) for e in evs[-limit:]]

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        with self._lock:
            al = list(self._alerts)[-limit:]
        return [{"time": time.strftime("%H:%M:%S", time.localtime(a.wall)), "watch_id": a.watch_id,
                 "title": a.title, "detail": a.text} for a in al]

    def add_watch(self, kind: str, feed=None, subject=None, value=None, note="", once=False, person=None) -> Watch:
        if kind not in WATCH_KINDS:
            raise ValueError(f"unknown watch kind {kind!r}; valid: {sorted(WATCH_KINDS)}")
        fid = self._parse_feed(feed)
        if feed is not None and fid not in self.feeds:
            raise ValueError(f"no such feed {feed!r}")
        if kind in ("count_at_least", "count_at_most", "dwell_exceeds") and value is None:
            raise ValueError(f"{kind} needs a numeric value")
        pid = self.parse_person(person) if person else None
        if person and pid is None:
            raise ValueError(f"no remembered person {person!r}")
        w = Watch(next(self._watch_ids), kind, fid, self._parse_subject(subject), value, note, once, person=pid)
        with self._lock:
            self.watches[w.id] = w
        return w

    def cancel_watch(self, wid: int) -> bool:
        with self._lock:
            return self.watches.pop(int(wid), None) is not None

    def set_tracking(self, ref, active: bool) -> list[str]:
        done = []
        for f in self._feeds(ref):
            (f.start_tracking if active else f.stop_tracking)()
            done.append(f.name)
        return done

    MESH_LAYERS = {"web", "mesh", "contours", "eyes", "gaze"}

    def display_state(self) -> dict:
        d = self.display.as_dict() if self.display else {}
        return {"overlay_style": d.pop("style", None), "layers": d,
                "dense_landmarks_per_feed": {f.name: f.settings.landmarks for f in self.feeds.values()}}

    def set_layer(self, name: str, enabled: bool) -> dict:
        if self.display is None or name not in self.display_help:
            raise ValueError(f"unknown layer {name!r}; valid: {sorted(self.display_help)}")
        setattr(self.display, name, bool(enabled))
        note = None
        if enabled and name in self.MESH_LAYERS:
            for f in self.feeds.values():
                if not f.settings.landmarks:
                    f.settings.landmarks = True
                    note = "dense landmarks enabled on all feeds"
        self.ui_actions.append({"action": "refresh"})
        return {"layer": name, "enabled": bool(enabled), **({"note": note} if note else {})}

    def set_style(self, style: str) -> dict:
        if style not in ("samaritan", "machine"):
            raise ValueError("style must be 'samaritan' or 'machine'")
        self.display.style = style
        self.ui_actions.append({"action": "refresh"})
        return {"overlay_style": style}

    def set_analysis(self, ref, small_faces=None, landmarks=None, threshold=None) -> list[dict]:
        out = []
        for f in self._feeds(ref):
            if small_faces is not None:
                f.settings.small_faces = bool(small_faces)
            if landmarks is not None:
                f.settings.landmarks = bool(landmarks)
            if threshold is not None:
                f.settings.threshold = float(min(0.95, max(0.2, threshold)))
            out.append({"feed": f.name, "small_face_mode": f.settings.small_faces,
                        "dense_landmarks": f.settings.landmarks, "detection_threshold": f.settings.threshold})
        return out

    def set_focus(self, feed_ref, subject_ref) -> bool:
        fid, sid = self._parse_feed(feed_ref), self._parse_subject(subject_ref)
        if subject_ref is None:
            self.focus = None
            return True
        if fid is None:
            for f in self.feeds.values():
                r = f.latest()
                if r and any(t.id == sid for t in r.tracks):
                    fid = f.id
                    break
        if fid is None or fid not in self.feeds:
            return False
        self.focus = (fid, sid)
        self.ui_actions.append({"action": "focus", "feed": fid, "subject": sid})
        return True
