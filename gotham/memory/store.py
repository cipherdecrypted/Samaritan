from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..inference.models import RecogniserProfile, recogniser_profile
from . import secure

YAW_BINS = (-90, -35, -12, 12, 35, 91)
PER_BIN = 3
VISIT_GAP = 60.0


def default_dir() -> Path:
    if os.environ.get("GOTHAM_MEMORY_DIR"):
        return Path(os.environ["GOTHAM_MEMORY_DIR"])
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "Gotham" / "memory"


def yaw_bin(yaw: float) -> int:
    for i in range(len(YAW_BINS) - 1):
        if YAW_BINS[i] <= yaw < YAW_BINS[i + 1]:
            return i
    return 2


@dataclass
class Person:
    id: int
    label: str
    created: float
    last_seen: float
    first_seen: float
    sightings: int
    exemplars: int
    bins: tuple
    thumb_q: float = 0.0
    alert: bool = False
    notes: str = ""

    @property
    def code(self) -> str:
        return f"P-{self.id:04d}"

    @property
    def name(self) -> str:
        return self.label or self.code


class MemoryStore:
    def __init__(self, path: Path | None = None, profile: RecogniserProfile | None = None):
        self.th = profile or recogniser_profile()
        self.reset_reason = ""
        d = Path(path) if path else default_dir()
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / "memory.db"
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS persons(
                id INTEGER PRIMARY KEY, label TEXT DEFAULT '', created REAL, first_seen REAL, last_seen REAL,
                thumb BLOB, thumb_q REAL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS exemplars(
                id INTEGER PRIMARY KEY, person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                emb BLOB, yaw REAL, quality REAL, created REAL);
            CREATE TABLE IF NOT EXISTS sightings(
                id INTEGER PRIMARY KEY, person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                kind TEXT, source TEXT, start REAL, end REAL, frames INTEGER, score REAL, thumb BLOB);
            CREATE INDEX IF NOT EXISTS ix_ex_person ON exemplars(person_id);
            CREATE INDEX IF NOT EXISTS ix_si_person ON sightings(person_id, start);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            -- possible sightings: someone resembled this person but not convincingly enough to be linked
            CREATE TABLE IF NOT EXISTS candidates(
                id INTEGER PRIMARY KEY, person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                other_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
                source TEXT, t REAL, score REAL, label TEXT, thumb BLOB, status TEXT DEFAULT 'open');
            CREATE INDEX IF NOT EXISTS ix_ca_person ON candidates(person_id, status);
            CREATE INDEX IF NOT EXISTS ix_si_source ON sightings(source, start);
        """)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(exemplars)")}
        if "thumb" not in cols:
            self.db.execute("ALTER TABLE exemplars ADD COLUMN thumb BLOB")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(persons)")}
        for col, ddl in (("notes", "TEXT DEFAULT ''"), ("mesh", "BLOB"), ("mesh_q", "REAL DEFAULT 0"),
                         ("alert", "INTEGER DEFAULT 0")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE persons ADD COLUMN {col} {ddl}")
        row = self.db.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        n_ex = self.db.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0]
        self.migrate_from = ""
        if row is None or row[0] != self.th.name:
            old = row[0] if row else "sface_2021dec"
            photos = self.db.execute("SELECT (SELECT COUNT(*) FROM exemplars WHERE thumb IS NOT NULL) + "
                                     "(SELECT COUNT(*) FROM persons WHERE thumb IS NOT NULL)").fetchone()[0]
            if n_ex and photos and not os.environ.get("GOTHAM_NO_MIGRATE"):
                self.migrate_from = old
            else:
                if n_ex or (row and row[0] != self.th.name):
                    self.reset_reason = f"recognition model changed ({old} -> {self.th.name})"
                    self.db.executescript("DELETE FROM candidates; DELETE FROM sightings; DELETE FROM exemplars; "
                                          "DELETE FROM persons;")
                self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('embed_model', ?)", (self.th.name,))
        self._enabled = True
        self.migration_progress: float | None = None
        self.version = 0
        self.emb_version = 0
        self._dup_cache: tuple | None = None
        self._alert_ids = {r[0] for r in self.db.execute("SELECT id FROM persons WHERE alert=1")}
        audited = self.db.execute("SELECT value FROM meta WHERE key='views_audit'").fetchone() is not None
        self.needs_audit = (not audited and not self.migrate_from and any(d.glob("memory.before-*.db")))
        if not audited and not self.needs_audit and not self.migrate_from:
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('views_audit', '2')")
        self.stray_removed = 0
        if self.db.execute("SELECT value FROM meta WHERE key='visits_v1'").fetchone() is None:
            self.compact_visits()
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('visits_v1', '1')")
        self._rebuild()
        if not self.migrate_from and not self.needs_audit:
            self.stray_removed = len(self.remove_stray_views())

    @property
    def enabled(self) -> bool:
        return self._enabled and not self.migrate_from and not self.needs_audit

    @enabled.setter
    def enabled(self, on: bool):
        self._enabled = bool(on)

    def _rebuild(self):
        with self._lock:
            rows = [] if self.migrate_from else self.db.execute("SELECT id, person_id, emb FROM exemplars").fetchall()
            if rows:
                self._eid = np.array([r[0] for r in rows], np.int64)
                self._pid = np.array([r[1] for r in rows], np.int64)
                self._mat = np.stack([np.frombuffer(secure.unprotect(r[2]), np.float32) for r in rows])
            else:
                self._eid = np.zeros(0, np.int64)
                self._pid = np.zeros(0, np.int64)
                self._mat = np.zeros((0, self.th.dim), np.float32)
            self._groups = None
            self.version += 1
            self.emb_version += 1

    def _append(self, eid: int, pid: int, emb: np.ndarray):
        self._eid = np.append(self._eid, np.int64(eid))
        self._pid = np.append(self._pid, np.int64(pid))
        self._mat = np.vstack([self._mat, np.asarray(emb, np.float32).reshape(1, -1)])
        self._groups = None
        self.emb_version += 1

    def _drop_rows(self, mask: np.ndarray):
        keep = ~mask
        self._eid, self._pid, self._mat = self._eid[keep], self._pid[keep], self._mat[keep]
        self._groups = None
        self.emb_version += 1

    def _group_index(self):
        if self._groups is None:
            order = np.argsort(self._pid, kind="stable")
            sp = self._pid[order]
            starts = np.flatnonzero(np.r_[True, sp[1:] != sp[:-1]]) if len(sp) else np.zeros(0, np.intp)
            self._groups = (sp[starts], order, starts)
        return self._groups

    def match(self, embs: np.ndarray, exclude: set[int] | None = None,
              weights: np.ndarray | None = None) -> list[tuple[int, float]]:
        embs = np.atleast_2d(embs)
        w = np.ones(len(embs)) if weights is None else np.asarray(weights, np.float64)
        w = w / max(w.sum(), 1e-9)
        with self._lock:
            if not len(self._pid):
                return []
            sims = embs @ self._mat.T
            uniq, order, starts = self._group_index()
        best = np.maximum.reduceat(sims[:, order], starts, axis=1)
        scores = w @ best
        order = np.argsort(-scores)
        return [(int(uniq[i]), float(scores[i])) for i in order if not (exclude and int(uniq[i]) in exclude)]

    def sample_score(self, pid: int, emb: np.ndarray) -> float:
        with self._lock:
            m = self._mat[self._pid == pid]
        return float((m @ emb).max()) if len(m) else -1.0

    @staticmethod
    def _jpeg(img: np.ndarray | None) -> bytes | None:
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return secure.protect(buf.tobytes()) if ok else None

    def create_person(self, samples: list[tuple[np.ndarray, float, float]], thumb: np.ndarray | None,
                      thumb_q: float, t: float) -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO persons(created, first_seen, last_seen, thumb, thumb_q) VALUES(?,?,?,?,?)",
                                  (time.time(), t, t, self._jpeg(thumb), thumb_q))
            pid = cur.lastrowid
            for s in samples:
                emb, yaw, q = s[:3]
                self._insert_exemplar(pid, emb, yaw, q, s[3] if len(s) > 3 else None)
            self.version += 1
            return pid

    def _insert_exemplar(self, pid, emb, yaw, q, thumb=None):
        if not np.isfinite(emb).all():
            return
        cur = self.db.execute("INSERT INTO exemplars(person_id, emb, yaw, quality, created, thumb) "
                              "VALUES(?,?,?,?,?,?)",
                              (pid, secure.protect(np.asarray(emb, np.float32).tobytes()), float(yaw), float(q),
                               time.time(), self._jpeg(thumb)))
        self._append(cur.lastrowid, pid, emb)

    def learn(self, pid: int, emb: np.ndarray, yaw: float, q: float, thumb: np.ndarray | None = None) -> bool:
        if not np.isfinite(emb).all():
            return False
        b = yaw_bin(yaw)
        lo, hi = YAW_BINS[b], YAW_BINS[b + 1]
        with self._lock:
            rows = self.db.execute("SELECT id, quality FROM exemplars WHERE person_id=? AND yaw>=? AND yaw<? "
                                   "ORDER BY quality", (pid, lo, hi)).fetchall()
            if len(rows) >= PER_BIN:
                if q <= rows[0][1] + 0.05:
                    return False
                self.db.execute("DELETE FROM exemplars WHERE id=?", (rows[0][0],))
                self._drop_rows(self._eid == rows[0][0])
            self._insert_exemplar(pid, emb, yaw, q, thumb)
            self.version += 1
            return True

    def update_thumb(self, pid: int, thumb: np.ndarray, q: float):
        with self._lock:
            cur = self.db.execute("SELECT thumb_q FROM persons WHERE id=?", (pid,)).fetchone()
            if cur and q > (cur[0] or 0) + 0.03:
                self.db.execute("UPDATE persons SET thumb=?, thumb_q=? WHERE id=?", (self._jpeg(thumb), q, pid))
                self.version += 1

    def add_sighting(self, pid: int, kind: str, source: str, start: float, end: float, frames: int,
                     score: float, thumb: np.ndarray | None) -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO sightings(person_id, kind, source, start, end, frames, score, thumb) "
                                  "VALUES(?,?,?,?,?,?,?,?)",
                                  (pid, kind, source, start, end, frames, score, self._jpeg(thumb)))
            self.db.execute("UPDATE persons SET last_seen=MAX(last_seen, ?), first_seen=MIN(first_seen, ?) WHERE id=?",
                            (end, start, pid))
            self.version += 1
            return cur.lastrowid

    def update_sighting(self, sid: int, end: float, frames: int, score: float, thumb: np.ndarray | None = None):
        with self._lock:
            if thumb is not None:
                self.db.execute("UPDATE sightings SET end=MAX(end, ?), frames=?, score=?, thumb=? WHERE id=?",
                                (end, frames, score, self._jpeg(thumb), sid))
            else:
                self.db.execute("UPDATE sightings SET end=MAX(end, ?), frames=?, score=? WHERE id=?",
                                (end, frames, score, sid))
            self.db.execute("UPDATE persons SET last_seen=MAX(last_seen, ?) WHERE id=(SELECT person_id FROM sightings "
                            "WHERE id=?)", (end, sid))
            self.version += 1

    def resume_visit(self, pid: int, source: str, start: float) -> dict | None:
        with self._lock:
            r = self.db.execute("SELECT id, frames, score FROM sightings WHERE person_id=? AND source=? AND kind='live' "
                                "AND end >= ? ORDER BY end DESC LIMIT 1", (pid, source, start - VISIT_GAP)).fetchone()
        return {"id": r[0], "frames": r[1] or 0, "score": r[2] or 0.0} if r else None

    def compact_visits(self, gap: float = VISIT_GAP) -> int:
        with self._lock:
            rows = self.db.execute("SELECT id, person_id, source, start, end, frames, score, thumb IS NOT NULL "
                                   "FROM sightings WHERE kind='live' ORDER BY person_id, source, start").fetchall()
            merged = 0
            cur = None
            for sid, pid, src, start, end, frames, score, has_thumb in rows:
                if cur and cur["pid"] == pid and cur["src"] == src and start - cur["end"] <= gap:
                    cur["end"] = max(cur["end"], end)
                    cur["frames"] += frames or 0
                    cur["score"] = max(cur["score"], score or 0.0)
                    if not cur["thumb"] and has_thumb:
                        self.db.execute("UPDATE sightings SET thumb=(SELECT thumb FROM sightings WHERE id=?) WHERE id=?",
                                        (sid, cur["id"]))
                        cur["thumb"] = True
                    cur["drop"].append(sid)
                    continue
                if cur and cur["drop"]:
                    merged += self._write_visit(cur)
                cur = {"id": sid, "pid": pid, "src": src, "end": end, "frames": frames or 0, "score": score or 0.0,
                       "thumb": bool(has_thumb), "drop": []}
            if cur and cur["drop"]:
                merged += self._write_visit(cur)
            if merged:
                self.version += 1
            return merged

    def _write_visit(self, v: dict) -> int:
        self.db.execute("UPDATE sightings SET end=?, frames=?, score=? WHERE id=?", (v["end"], v["frames"], v["score"], v["id"]))
        self.db.executemany("DELETE FROM sightings WHERE id=?", [(x,) for x in v["drop"]])
        return len(v["drop"])

    def associates(self, pid: int, k: int = 6) -> list[dict]:
        with self._lock:
            rows = self.db.execute("""
                SELECT s2.person_id, COUNT(DISTINCT s1.id), SUM(MIN(s1.end, s2.end) - MAX(s1.start, s2.start))
                FROM sightings s1 JOIN sightings s2
                  ON s2.source = s1.source AND s2.person_id != s1.person_id AND s2.start < s1.end AND s2.end > s1.start
                WHERE s1.person_id = ? AND s1.kind = 'live' AND s2.kind = 'live'
                GROUP BY s2.person_id ORDER BY 3 DESC LIMIT ?""", (pid, k)).fetchall()
        return [{"id": r[0], "visits": r[1], "seconds": float(r[2] or 0)} for r in rows]

    def touch(self, pid: int, t: float):
        with self._lock:
            self.db.execute("UPDATE persons SET last_seen=MAX(last_seen, ?) WHERE id=?", (t, pid))

    def rename(self, pid: int, label: str):
        with self._lock:
            self.db.execute("UPDATE persons SET label=? WHERE id=?", (label.strip()[:40], pid))
            self.version += 1

    def forget(self, pid: int):
        with self._lock:
            self.db.execute("DELETE FROM persons WHERE id=?", (pid,))
            self._drop_rows(self._pid == pid)
            self._alert_ids.discard(pid)
            self.version += 1

    def merge(self, keep: int, drop: int):
        if keep == drop:
            return
        with self._lock:
            self.db.execute("UPDATE exemplars SET person_id=? WHERE person_id=?", (keep, drop))
            self._pid[self._pid == drop] = keep
            self._groups = None
            self.emb_version += 1
            self.db.execute("UPDATE sightings SET person_id=? WHERE person_id=?", (keep, drop))
            self.db.execute("UPDATE candidates SET person_id=? WHERE person_id=?", (keep, drop))
            self.db.execute("UPDATE candidates SET other_id=? WHERE other_id=?", (keep, drop))
            self.db.execute("DELETE FROM candidates WHERE person_id=other_id")
            a = self.db.execute("SELECT first_seen, last_seen, thumb_q, label, COALESCE(notes, ''), thumb, mesh, "
                                "COALESCE(mesh_q, 0), COALESCE(alert, 0) FROM persons WHERE id=?", (drop,)).fetchone()
            if a:
                self.db.execute("UPDATE persons SET first_seen=MIN(first_seen, ?), last_seen=MAX(last_seen, ?), "
                                "label=CASE WHEN label='' THEN ? ELSE label END, "
                                "notes=CASE WHEN COALESCE(notes, '')='' THEN ? WHEN ?='' THEN notes "
                                "ELSE notes || char(10) || ? END, alert=MAX(COALESCE(alert, 0), ?) WHERE id=?",
                                (a[0], a[1], a[3], a[4], a[4], a[4], a[8], keep))
                self.db.execute("UPDATE persons SET thumb=?, thumb_q=? WHERE id=? AND COALESCE(thumb_q, 0) < ?",
                                (a[5], a[2], keep, a[2] or 0))
                self.db.execute("UPDATE persons SET mesh=?, mesh_q=? WHERE id=? AND COALESCE(mesh_q, 0) < ?",
                                (a[6], a[7], keep, a[7]))
                if a[8]:
                    self._alert_ids.add(keep)
            self._alert_ids.discard(drop)
            self.db.execute("DELETE FROM persons WHERE id=?", (drop,))
            rows = self.db.execute("SELECT id, yaw, quality FROM exemplars WHERE person_id=? ORDER BY quality DESC",
                                   (keep,)).fetchall()
            counts: dict[int, int] = {}
            for eid, yaw, _ in rows:
                b = yaw_bin(yaw)
                counts[b] = counts.get(b, 0) + 1
                if counts[b] > PER_BIN:
                    self.db.execute("DELETE FROM exemplars WHERE id=?", (eid,))
                    self._drop_rows(self._eid == eid)
            self.version += 1

    def clear(self):
        with self._lock:
            self.db.executescript("DELETE FROM candidates; DELETE FROM sightings; DELETE FROM exemplars; "
                                  "DELETE FROM persons;")
            self.db.execute("VACUUM")
            self._alert_ids.clear()
            self._rebuild()

    def persons(self) -> list[Person]:
        with self._lock:
            rows = self.db.execute("""
                SELECT p.id, p.label, p.created, p.last_seen, p.first_seen, COALESCE(s.n, 0), COALESCE(e.n, 0),
                       COALESCE(p.thumb_q, 0), COALESCE(p.alert, 0)
                FROM persons p
                LEFT JOIN (SELECT person_id, COUNT(*) n FROM sightings GROUP BY person_id) s ON s.person_id = p.id
                LEFT JOIN (SELECT person_id, COUNT(*) n FROM exemplars GROUP BY person_id) e ON e.person_id = p.id
                ORDER BY p.last_seen DESC""").fetchall()
            bins = {}
            for pid, yaw in self.db.execute("SELECT person_id, yaw FROM exemplars"):
                bins.setdefault(pid, set()).add(yaw_bin(yaw))
        return [Person(r[0], r[1] or "", r[2], r[3] or r[2], r[4] or r[2], r[5], r[6], tuple(sorted(bins.get(r[0], ()))),
                       r[7], bool(r[8])) for r in rows]

    def person(self, pid: int) -> Person | None:
        with self._lock:
            r = self.db.execute("SELECT id, label, created, last_seen, first_seen, COALESCE(thumb_q, 0), "
                                "COALESCE(alert, 0), COALESCE(notes, '') FROM persons WHERE id=?", (pid,)).fetchone()
            if r is None:
                return None
            n_s = self.db.execute("SELECT COUNT(*) FROM sightings WHERE person_id=?", (pid,)).fetchone()[0]
            yaws = [y for (y,) in self.db.execute("SELECT yaw FROM exemplars WHERE person_id=?", (pid,))]
        return Person(r[0], r[1] or "", r[2], r[3] or r[2], r[4] or r[2], n_s, len(yaws),
                      tuple(sorted({yaw_bin(y) for y in yaws})), r[5], bool(r[6]), r[7])

    def thumb(self, pid: int) -> bytes | None:
        with self._lock:
            r = self.db.execute("SELECT thumb FROM persons WHERE id=?", (pid,)).fetchone()
        return secure.unprotect(r[0]) if r and r[0] else None

    def sightings(self, pid: int, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT id, kind, source, start, end, frames, score, thumb FROM sightings "
                                   "WHERE person_id=? ORDER BY start DESC LIMIT ?", (pid, limit)).fetchall()
        return [{"id": r[0], "kind": r[1], "source": r[2], "start": r[3], "end": r[4], "frames": r[5], "score": r[6],
                 "thumb": secure.unprotect(r[7]) if r[7] else None} for r in rows]

    def recent_sightings(self, since: float) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT person_id, kind, source, start, end, frames, score FROM sightings "
                                   "WHERE end>=? ORDER BY start DESC", (since,)).fetchall()
        return [{"person_id": r[0], "kind": r[1], "source": r[2], "start": r[3], "end": r[4], "frames": r[5],
                 "score": r[6]} for r in rows]

    def views(self, pid: int) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT yaw, quality, thumb, id FROM exemplars WHERE person_id=? ORDER BY yaw",
                                   (pid,)).fetchall()
        return [{"id": r[3], "yaw": r[0], "quality": r[1], "thumb": secure.unprotect(r[2]) if r[2] else None} for r in rows]

    def profile_similarity(self, a: int, b: int) -> float:
        with self._lock:
            A, B = self._mat[self._pid == a], self._mat[self._pid == b]
        return float((A @ B.T).max()) if len(A) and len(B) else -1.0

    def similar(self, pid: int, k: int = 3) -> list[tuple[int, float]]:
        with self._lock:
            mine = self._mat[self._pid == pid]
            if not len(mine):
                return []
            uniq, order, starts = self._group_index()
            sims = mine @ self._mat.T
        best = np.maximum.reduceat(sims[:, order], starts, axis=1).max(axis=0)
        best[uniq == pid] = -np.inf
        top = np.argsort(-best)[:k]
        return [(int(uniq[i]), float(best[i])) for i in top if np.isfinite(best[i])]

    def _dismissed(self) -> set[tuple[int, int]]:
        r = self.db.execute("SELECT value FROM meta WHERE key='dismissed_pairs'").fetchone()
        if not r or not r[0]:
            return set()
        return {tuple(int(x) for x in p.split("-")) for p in r[0].split(",") if "-" in p}

    def dismiss_pair(self, a: int, b: int):
        with self._lock:
            d = self._dismissed() | {(min(a, b), max(a, b))}
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('dismissed_pairs', ?)",
                            (",".join(f"{x}-{y}" for x, y in sorted(d)),))
            self.version += 1

    def duplicate_pairs(self, threshold: float | None = None, limit: int = 20) -> list[tuple[int, int, float]]:
        thr = self.th.match - 0.04 if threshold is None else threshold
        with self._lock:
            if len(self._pid) < 2:
                return []
            dismissed = self._dismissed()
            key = (self.emb_version, thr, limit, tuple(sorted(dismissed)))
            if self._dup_cache is not None and self._dup_cache[0] == key:
                return list(self._dup_cache[1])
            mat, pids = self._mat, self._pid
        best: dict[tuple[int, int], float] = {}
        for r0 in range(0, len(pids), 1024):
            sims = mat[r0:r0 + 1024] @ mat.T
            li, jj = np.nonzero(sims >= thr)
            keep = jj > li + r0
            li, jj = li[keep], jj[keep]
            for i, j, v in zip((li + r0).tolist(), jj.tolist(), sims[li, jj].tolist()):
                a, b = int(pids[i]), int(pids[j])
                if a == b:
                    continue
                k = (min(a, b), max(a, b))
                if k in dismissed:
                    continue
                best[k] = max(best.get(k, -1.0), v)
        out = sorted(((a, b, s) for (a, b), s in best.items()), key=lambda x: -x[2])[:limit]
        with self._lock:
            self._dup_cache = (key, out)
        return list(out)

    @property
    def certain_duplicate(self) -> float:
        return self.th.match + 0.2

    @property
    def certain_whole(self) -> float:
        return self.th.match + 0.05

    def duplicate_review(self, limit: int = 10_000) -> list[tuple[int, int, float, float]]:
        out = []
        for a, b, sc in self.duplicate_pairs(limit=limit):
            whole = self.pair_score(a, b)
            if whole >= self.th.new:
                out.append((a, b, sc, whole))
        return out

    def is_certain(self, sc: float, whole: float) -> bool:
        return sc >= self.certain_duplicate and whole >= self.certain_whole

    def merge_certain_duplicates(self) -> list[tuple[int, int, float]]:
        parent: dict[int, int] = {}

        def root(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x
        done = []
        for a, b, sc in self.duplicate_pairs(threshold=self.certain_duplicate, limit=10_000):
            if self.pair_score(a, b) < self.certain_whole:
                continue
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            keep, drop = min(ra, rb), max(ra, rb)
            self.merge(keep, drop)
            parent[drop] = keep
            done.append((keep, drop, sc))
        return done

    STRAY_BELOW = 0.15

    def view_report(self, pid: int) -> list[dict]:
        with self._lock:
            sel = self._pid == pid
            ids, m = self._eid[sel], self._mat[sel]
        n = len(ids)
        if n < 2:
            return [{"id": int(e), "agreement": None, "stray": False} for e in ids]
        g = m @ m.T
        np.fill_diagonal(g, np.nan)
        agree = np.nanmean(g, axis=1)
        out = []
        for i in range(n):
            stray = False
            if n >= 3 and agree[i] < self.STRAY_BELOW:
                rest = np.delete(np.delete(g, i, 0), i, 1)
                stray = bool(np.nanmean(rest) >= self.th.match)
            out.append({"id": int(ids[i]), "agreement": float(agree[i]), "stray": stray})
        return out

    def stray_views(self) -> list[tuple[int, int]]:
        with self._lock:
            pids = np.unique(self._pid)
        out = []
        for pid in pids:
            out += [(v["id"], int(pid)) for v in self.view_report(int(pid)) if v["stray"]]
        return out

    def remove_view(self, eid: int) -> bool:
        with self._lock:
            r = self.db.execute("SELECT person_id FROM exemplars WHERE id=?", (eid,)).fetchone()
            if r is None or self.db.execute("SELECT COUNT(*) FROM exemplars WHERE person_id=?", (r[0],)).fetchone()[0] < 2:
                return False
            self.db.execute("DELETE FROM exemplars WHERE id=?", (eid,))
            self._drop_rows(self._eid == eid)
            self.version += 1
            return True

    def remove_stray_views(self) -> list[tuple[int, int]]:
        removed = []
        for _ in range(3):
            stray = self.stray_views()
            if not stray:
                break
            for eid, pid in stray:
                if self.remove_view(eid):
                    removed.append((eid, pid))
        return removed

    def pair_score(self, a: int, b: int) -> float:
        with self._lock:
            A, B = self._mat[self._pid == a], self._mat[self._pid == b]
        if not len(A) or not len(B):
            return -1.0
        s = A @ B.T
        return min(float(s.max(axis=1).mean()), float(s.max(axis=0).mean()))

    def prunable(self, older_than_days: float = 7.0) -> list[int]:
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            rows = self.db.execute("""
                SELECT p.id FROM persons p WHERE (p.label IS NULL OR p.label = '') AND p.last_seen < ?
                AND COALESCE(p.alert, 0) = 0 AND COALESCE(p.notes, '') = ''
                AND (SELECT COUNT(*) FROM sightings s WHERE s.person_id = p.id) <= 1""", (cutoff,)).fetchall()
        return [r[0] for r in rows]

    def prune(self, older_than_days: float = 7.0) -> int:
        ids = self.prunable(older_than_days)
        with self._lock:
            for pid in ids:
                self.db.execute("DELETE FROM persons WHERE id=?", (pid,))
            if ids:
                self._drop_rows(np.isin(self._pid, ids))
                self.version += 1
        return len(ids)

    def set_notes(self, pid: int, text: str):
        with self._lock:
            self.db.execute("UPDATE persons SET notes=? WHERE id=?", (text.strip()[:2000], pid))
            self.version += 1

    def set_alert(self, pid: int, on: bool):
        with self._lock:
            self.db.execute("UPDATE persons SET alert=? WHERE id=?", (1 if on else 0, pid))
            (self._alert_ids.add if on else self._alert_ids.discard)(pid)
            self.version += 1

    def alert_ids(self) -> set[int]:
        with self._lock:
            return set(self._alert_ids)

    def update_mesh(self, pid: int, mesh: np.ndarray, q: float) -> bool:
        with self._lock:
            cur = self.db.execute("SELECT COALESCE(mesh_q, 0) FROM persons WHERE id=?", (pid,)).fetchone()
            if cur is None or q <= cur[0] + 0.02:
                return False
            blob = secure.protect(np.asarray(mesh, np.float16).tobytes())
            self.db.execute("UPDATE persons SET mesh=?, mesh_q=? WHERE id=?", (blob, float(q), pid))
            self.version += 1
            return True

    def mesh(self, pid: int) -> np.ndarray | None:
        with self._lock:
            r = self.db.execute("SELECT mesh FROM persons WHERE id=?", (pid,)).fetchone()
        if not r or not r[0]:
            return None
        return np.frombuffer(secure.unprotect(r[0]), np.float16).astype(np.float32).reshape(-1, 3)

    def add_candidate(self, pid: int, source: str, t: float, score: float, label: str,
                      thumb: np.ndarray | None, other: int | None = None) -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO candidates(person_id, other_id, source, t, score, label, thumb) "
                                  "VALUES(?,?,?,?,?,?,?)", (pid, other, source, t, float(score), label, self._jpeg(thumb)))
            self.version += 1
            return cur.lastrowid

    def update_candidate(self, cid: int, score: float | None = None, thumb: np.ndarray | None = None,
                         other: int | None = None):
        with self._lock:
            if score is not None:
                self.db.execute("UPDATE candidates SET score=MAX(score, ?) WHERE id=?", (float(score), cid))
            if thumb is not None:
                self.db.execute("UPDATE candidates SET thumb=? WHERE id=?", (self._jpeg(thumb), cid))
            if other is not None:
                self.db.execute("UPDATE candidates SET other_id=? WHERE id=?", (other, cid))
            self.version += 1

    def delete_candidate(self, cid: int):
        with self._lock:
            self.db.execute("DELETE FROM candidates WHERE id=?", (cid,))
            self.version += 1

    def candidate(self, cid: int) -> dict | None:
        with self._lock:
            r = self.db.execute("SELECT id, person_id, other_id, source, t, score, label, thumb, status "
                                "FROM candidates WHERE id=?", (cid,)).fetchone()
        if r is None:
            return None
        return {"id": r[0], "person_id": r[1], "other_id": r[2], "source": r[3], "t": r[4], "score": r[5],
                "label": r[6], "thumb": secure.unprotect(r[7]) if r[7] else None, "status": r[8]}

    def candidates(self, pid: int, limit: int = 60) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT id, other_id, source, t, score, label, thumb FROM candidates "
                                   "WHERE person_id=? AND status='open' ORDER BY score DESC, t DESC LIMIT ?",
                                   (pid, limit)).fetchall()
        return [{"id": r[0], "other_id": r[1], "source": r[2], "t": r[3], "score": r[4], "label": r[5],
                 "thumb": secure.unprotect(r[6]) if r[6] else None} for r in rows]

    def resolve_candidate(self, cid: int, confirm: bool) -> str:
        c = self.candidate(cid)
        if c is None:
            return "gone"
        pid, other = c["person_id"], c["other_id"]
        if confirm:
            if other is not None and other != pid and self.person(other) is not None:
                self.merge(min(pid, other), max(pid, other))
                return "merged"
            with self._lock:
                self.db.execute("INSERT INTO sightings(person_id, kind, source, start, end, frames, score, thumb) "
                                "SELECT person_id, 'confirmed', source, t, t, 1, score, thumb FROM candidates WHERE id=?",
                                (cid,))
                self.db.execute("UPDATE persons SET last_seen=MAX(last_seen, ?), first_seen=MIN(first_seen, ?) "
                                "WHERE id=?", (c["t"], c["t"], pid))
                self.db.execute("DELETE FROM candidates WHERE id=?", (cid,))
                self.version += 1
            return "confirmed"
        with self._lock:
            self.db.execute("UPDATE candidates SET status='rejected' WHERE id=?", (cid,))
            self.version += 1
        if other is not None and other != pid:
            self.dismiss_pair(pid, other)
        return "rejected"

    def sighting_rows(self, pid: int, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT id, kind, source, start, end, frames, score, COALESCE(length(thumb), 0) "
                                   "FROM sightings WHERE person_id=? ORDER BY start DESC LIMIT ?", (pid, limit)).fetchall()
        return [{"id": r[0], "kind": r[1], "source": r[2], "start": r[3], "end": r[4], "frames": r[5],
                 "score": r[6] or 0.0, "tsize": r[7]} for r in rows]

    def sighting_thumb(self, sid: int) -> bytes | None:
        with self._lock:
            r = self.db.execute("SELECT thumb FROM sightings WHERE id=?", (sid,)).fetchone()
        return secure.unprotect(r[0]) if r and r[0] else None

    def signature(self, pid: int) -> np.ndarray | None:
        with self._lock:
            m = self._mat[self._pid == pid]
        if not len(m):
            return None
        v = m.mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def profile_stats(self, pid: int) -> dict:
        with self._lock:
            rows = self.db.execute("SELECT source, kind, start, end, score FROM sightings WHERE person_id=?",
                                   (pid,)).fetchall()
            ex = self.db.execute("SELECT yaw, quality FROM exemplars WHERE person_id=?", (pid,)).fetchall()
            m = self._mat[self._pid == pid]
            n_cand = self.db.execute("SELECT COUNT(*) FROM candidates WHERE person_id=? AND status='open'",
                                     (pid,)).fetchone()[0]
        total = 0.0
        sources: dict[str, dict] = {}
        hours = [0.0] * 24
        today = time.localtime()
        day0 = time.mktime((today.tm_year, today.tm_mon, today.tm_mday, 0, 0, 0, 0, 0, -1))
        days = [0] * 14
        scores = []
        for src, kind, start, end, score in rows:
            dur = max(0.0, (end or start) - start)
            total += dur
            e = sources.setdefault(src or "?", {"source": src or "?", "n": 0, "seconds": 0.0, "last": 0.0})
            e["n"] += 1
            e["seconds"] += dur
            e["last"] = max(e["last"], end or start)
            t, left = start, max(dur, 1.0)
            for _ in range(48):
                lt = time.localtime(t)
                to_next = 3600 - (lt.tm_min * 60 + lt.tm_sec)
                step = min(left, to_next)
                hours[lt.tm_hour] += step
                left -= step
                t += step
                if left <= 0:
                    break
            d = int((day0 - start) // 86400) + 1 if start < day0 else 0
            if 0 <= d < 14:
                days[13 - d] += 1
            if score:
                scores.append(score)
        coherence = None
        if len(m) >= 2:
            g = m @ m.T
            coherence = float((g.sum() - np.trace(g)) / (len(m) * (len(m) - 1)))
        bins = [0] * (len(YAW_BINS) - 1)
        for yaw, _ in ex:
            bins[yaw_bin(yaw)] += 1
        return {"sightings": len(rows), "total_seconds": total, "sources": sorted(sources.values(), key=lambda x: -x["n"]),
                "hours": hours, "days": days, "best_score": max(scores) if scores else None,
                "mean_score": float(np.mean(scores)) if scores else None, "views": len(ex), "bins": bins,
                "coherence": coherence, "candidates": n_cand}

    MIN_VIEW_Q = 0.4

    def _remake(self, blobs: list, embedder, detector, progress, lo: float, hi: float) -> list:
        from ..inference.embedder import align112
        found = []
        for i, blob in enumerate(blobs):
            found.append(self._face_in_photo(blob, detector))
            if progress and i % 25 == 0:
                progress(lo + (hi - lo) * 0.8 * i / max(len(blobs), 1))
        idx = [i for i, (big, kps) in enumerate(found) if kps is not None]
        crops = [align112(found[i][0], found[i][1]) for i in idx]
        embs = []
        for j in range(0, len(crops), 64):
            embs += embedder.embed_crops(crops[j:j + 64], flip=True)
            if progress:
                progress(lo + (hi - lo) * (0.8 + 0.2 * min(1.0, (j + 64) / max(len(crops), 1))))
        out = [(None, 0.0, False)] * len(blobs)
        for i, (v, q) in zip(idx, embs):
            if np.isfinite(v).all():
                out[i] = (v, q, True)
        return out

    def _weak_views(self, rows: list, quality: list) -> list[int]:
        by_pid: dict[int, list] = {}
        for (eid, pid), (_, q, ok) in zip(rows, quality):
            by_pid.setdefault(pid, []).append((q if ok else -1.0, eid))
        drop = []
        for pid, views in by_pid.items():
            views.sort(reverse=True)
            drop += [eid for i, (q, eid) in enumerate(views) if i > 0 and q < self.MIN_VIEW_Q]
        return drop

    def migrate(self, embedder, detector, progress=None) -> dict:
        if not self.migrate_from:
            return {}
        t0 = time.time()
        user_progress = progress

        def progress(f):
            self.migration_progress = f
            if user_progress:
                user_progress(f)
        progress(0.0)
        with self._lock:
            safe = "".join(ch if ch.isalnum() else "_" for ch in self.migrate_from)[:60]
            backup = sqlite3.connect(str(self.path.with_name(f"memory.before-{safe}.db")))
            self.db.backup(backup)
            backup.close()
            rows = self.db.execute("SELECT id, person_id, thumb, quality FROM exemplars").fetchall()
            photos = self.db.execute("SELECT id, thumb, COALESCE(thumb_q, 0) FROM persons WHERE thumb IS NOT NULL"
                                     ).fetchall()
        views = self._remake([r[2] for r in rows], embedder, detector, progress, 0.0, 0.75)
        pics = self._remake([p[1] for p in photos], embedder, detector, progress, 0.75, 0.95)
        drop = set(self._weak_views([(r[0], r[1]) for r in rows], views))
        with self._lock:
            for (eid, pid, _, q_old), (v, q, ok) in zip(rows, views):
                if eid in drop or v is None:
                    self.db.execute("DELETE FROM exemplars WHERE id=?", (eid,))
                else:
                    self.db.execute("UPDATE exemplars SET emb=?, quality=? WHERE id=?",
                                    (secure.protect(np.asarray(v, np.float32).tobytes()), float(q_old or 0.3) * 0.75, eid))
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('embed_model', ?)", (self.th.name,))
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('views_audit', '2')")
            old, self.migrate_from, self.needs_audit = self.migrate_from, "", False
            self._dup_cache = None
            self._rebuild()
        for (pid, _, q_old), (v, q, ok) in zip(photos, pics):
            if ok and q >= self.MIN_VIEW_Q:
                try:
                    self.learn(pid, v, 0.0, max(0.3, float(q_old) * 0.75))
                except sqlite3.Error:
                    pass
        with self._lock:
            n_p = self.db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            n_e = self.db.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0]
            self.version += 1
        self.migration_progress = None
        return {"from": old, "to": self.th.name, "profiles": n_p, "views": n_e, "dropped": len(drop),
                "seconds": round(time.time() - t0, 1)}

    def audit_views(self, embedder, detector, progress=None) -> dict:
        if not self.needs_audit:
            return {}
        t0 = time.time()

        def prog(f):
            self.migration_progress = f
            if progress:
                progress(f)
        prog(0.0)
        with self._lock:
            rows = self.db.execute("SELECT id, person_id, thumb FROM exemplars").fetchall()
        quality = self._remake([r[2] for r in rows], embedder, detector, prog, 0.0, 1.0)
        drop = self._weak_views([(r[0], r[1]) for r in rows], quality)
        with self._lock:
            self.db.executemany("DELETE FROM exemplars WHERE id=?", [(e,) for e in drop])
            if drop:
                self._drop_rows(np.isin(self._eid, drop))
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('views_audit', '2')")
            self.needs_audit = False
            self._dup_cache = None
            self.version += 1
        self.migration_progress = None
        return {"views": len(rows), "dropped": len(drop), "seconds": round(time.time() - t0, 1)}

    @staticmethod
    def _face_in_photo(blob, detector):
        if not blob:
            return None, None
        img = cv2.imdecode(np.frombuffer(secure.unprotect(blob), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        dets = detector.detect(big, threshold=0.3)
        if not dets:
            return big, None
        c = np.array([big.shape[1] / 2, big.shape[0] / 2])
        d = min(dets, key=lambda d: np.linalg.norm(d.box[:2] + d.box[2:] / 2 - c))
        return big, d.kps.astype(np.float32)

    def stats(self) -> dict:
        with self._lock:
            n = self.db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            s = self.db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
        return {"profiles": n, "sightings": s, "exemplars": int(len(self._pid)), "path": str(self.path),
                "model": self.th.display}

    def close(self):
        with self._lock:
            self.db.close()
