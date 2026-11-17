from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMainWindow, QMessageBox, QSystemTrayIcon

from ..assistant.observer import Observer
from ..core.feed import Feed
from ..core.hub import Hub
from ..sources.capture import Kind, SourceSpec, Status
from ..sources.discovery import CameraInfo, discover_cameras
from .imagelab import ImageLab, ImageView
from .memory_panel import MemoryPanel
from .overlay import LAYER_HELP, Layers
from .viewport import FeedGrid, FeedView

WEB = Path(__file__).parent / "web"


class Bridge(QObject):
    actionReceived = Signal(dict)
    commandReceived = Signal(str)
    holeChanged = Signal(int, int, int, int)
    pageReady = Signal()
    bootFinished = Signal()

    @Slot(str)
    def action(self, payload: str):
        try:
            self.actionReceived.emit(json.loads(payload))
        except json.JSONDecodeError:
            pass

    @Slot(str)
    def command(self, text: str):
        self.commandReceived.emit(text)

    @Slot(int, int, int, int)
    def hole(self, x, y, w, h):
        self.holeChanged.emit(x, y, w, h)

    @Slot()
    def ready(self):
        self.pageReady.emit()

    @Slot()
    def booted(self):
        self.bootFinished.emit()


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if os.environ.get("GOTHAM_JS_DEBUG") or level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            print(f"[js:{line}] {message}", flush=True)


class Signals(QObject):
    boot = Signal(str, str, float)
    reply = Signal(str, str)
    status = Signal(str)
    error = Signal(str)
    cameras = Signal(list)
    labChanged = Signal()
    migrated = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("GOTHAM · SAMARITAN")
        self.resize(1680, 980)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
        self._said: dict = {}
        self._last_say = 0.0
        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(QApplication.windowIcon() or QIcon(), self)
            self.tray.setToolTip("Gotham")
            self.tray.show()
        self.feed_full = False
        QShortcut(QKeySequence("Esc"), self, self.exit_feed_full)
        self.hub = Hub()
        try:
            from ..memory.store import MemoryStore
            self.hub.memory = MemoryStore()
        except Exception as e:
            print("face memory unavailable:", e, flush=True)
        self.memory_panel = MemoryPanel(self.hub.memory)
        self.layers = Layers()
        self.hub.display, self.hub.display_help = self.layers, LAYER_HELP
        self.observer = Observer(self.hub)
        self.sig = Signals()
        self.mode = "live"
        self.selected: int | None = None
        self.cameras: list[CameraInfo] = []
        self.booted = False
        self._page_ready = False
        self._pending_js: list[str] = []
        self._log_buf: list[dict] = []
        self._proc = psutil.Process()
        psutil.cpu_percent(None)

        self.web = QWebEngineView(self)
        self.page = Page(self.web)
        self.web.setPage(self.page)
        s = self.page.settings()
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        s.setAttribute(QWebEngineSettings.ShowScrollBars, False)
        self.page.setBackgroundColor("#ffffff")
        self.bridge = Bridge()
        self.channel = QWebChannel(self.page)
        self.channel.registerObject("bridge", self.bridge)
        self.page.setWebChannel(self.channel)
        self.setCentralWidget(self.web)

        self.grid = FeedGrid(self.web)
        self.lab = ImageLab(on_change=self.sig.labChanged.emit, memory=self.hub.memory)
        self.imageview = ImageView(self.lab, self.layers, self.web)
        self.grid.hide()
        self.imageview.hide()
        self._hole = (0, 0, 0, 0)

        b = self.bridge
        b.actionReceived.connect(self.on_action)
        b.commandReceived.connect(self.on_command)
        b.holeChanged.connect(self.on_hole)
        b.pageReady.connect(self.on_page_ready)
        b.bootFinished.connect(self.on_booted)
        self.sig.boot.connect(self.on_boot_step)
        self.sig.reply.connect(self.on_reply)
        self.sig.status.connect(lambda t: self.js(f"G.status({json.dumps(t)})"))
        self.sig.error.connect(self.on_observer_error)
        self.sig.cameras.connect(self.on_cameras)
        self.sig.migrated.connect(self.on_memory_migrated)
        self.sig.labChanged.connect(self.on_lab_changed)
        self.imageview.faceClicked.connect(lambda k, sh: (self.lab.set_ab(k) if sh else self.lab.inspect(k)))
        self.observer.on_reply = lambda h, body: self.sig.reply.emit(h, body)
        self.observer.on_status = lambda t: self.sig.status.emit(t)
        self.observer.on_error = lambda m: self.sig.error.emit(m)

        self.render_timer = QTimer(self, interval=16, timeout=self.on_render)
        self.state_timer = QTimer(self, interval=250, timeout=self.push_state)
        self.tick_timer = QTimer(self, interval=200, timeout=self.on_tick)

        self.web.load(QUrl.fromLocalFile(str(WEB / "index.html")))

    def js(self, code: str):
        if not self._page_ready:
            self._pending_js.append(code)
        else:
            self.page.runJavaScript(code)

    def log(self, k: str, text: str = "", cls: str = ""):
        self._log_buf.append({"t": time.strftime("%H:%M:%S"), "k": k, "text": text, "cls": cls})

    def on_page_ready(self):
        self._page_ready = True
        self.memory_panel.page_reloaded()
        for c in self._pending_js:
            self.page.runJavaScript(c)
        self._pending_js.clear()
        if self.args.skip_boot:
            self.js("$('#boot').hide()")
            threading.Thread(target=self._boot_worker, args=(True,), daemon=True).start()
        else:
            self.js("G.bootStart()")
            threading.Thread(target=self._boot_worker, daemon=True).start()

    def on_hole(self, x, y, w, h):
        self._hole = (x, y, w, h)
        self._place_views()

    def _place_views(self):
        x, y, w, h = self._hole
        for wdg in (self.grid, self.imageview):
            if wdg is self.grid and self.feed_full:
                wdg.setGeometry(0, 0, self.web.width(), self.web.height())
            else:
                wdg.setGeometry(x, y, w, h)
            wdg.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "feed_full", False):
            self._place_views()

    def cycle_expand(self, fid: int):
        if self.grid.expanded != fid:
            self.grid.expanded, self.feed_full = None, False
            self.grid.toggle_expand(fid)
        elif not self.feed_full:
            self.set_feed_full(fid, True)
            return
        else:
            self.set_feed_full(fid, False)
            self.grid.toggle_expand(fid)
        self._place_views()

    def set_feed_full(self, fid: int | None, on: bool):
        self.feed_full = bool(on and fid is not None)
        if self.feed_full:
            self.grid.expanded = fid
            self.grid.setFocus()
        self._place_views()
        self.grid._layout()

    def exit_feed_full(self):
        if self.feed_full:
            self.set_feed_full(None, False)

    def _boot_worker(self, quiet=False):
        emit = self.sig.boot.emit
        steps = []
        t0 = time.time()
        emit("line", "x0.0/gotham", 0)
        emit("load", "resources.runtime.onnx", 5)
        from ..inference.runtime import shared_session
        from ..config import YUNET_MODEL
        from ..inference.models import make_detector, make_embedder
        s1 = shared_session(YUNET_MODEL)
        emit("line", f"+0.1/{'dml' if s1.on_gpu else 'cpu'}  {s1.provider}", 0)
        emit("load", "resources.models.gothamfaces", 20)
        import numpy as np
        s1.run(np.zeros((1, 3, 640, 640), np.float32))
        emit("load", "resources.models.face_landmarker", 35)
        from ..inference.landmarks import FaceLandmarks
        FaceLandmarks().close()
        emit("line", "[mesh] 478pt + iris + blendshapes", 0)
        emit("load", "resources.models.detector", 45)
        det = make_detector()
        det.detect(np.zeros((480, 640, 3), np.uint8))
        emit("line", f"[det] {det.profile.display}", 0)
        emit("load", "resources.models.recognition", 55)
        emb = make_embedder()
        emb.embed(np.zeros((112, 112, 3), np.uint8), np.array([[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]], np.float32))
        emit("line", f"[rec] {emb.profile.display}", 0)
        mem = self.hub.memory
        if mem is not None and mem.stray_removed:
            emit("line", f"[mem] removed {mem.stray_removed} stored view(s) that showed someone else", 0)
        if mem is not None and mem.reset_reason:
            emit("line", f"[mem] reset: {mem.reset_reason}", 0)
        if mem is not None and (mem.migrate_from or mem.needs_audit):
            emit("line", f"[mem] {'upgrading face memory to ' + emb.profile.display if mem.migrate_from else 'auditing stored views'}"
                         " (background)", 0)
            threading.Thread(target=self._migrate_memory, args=(mem, emb, det), daemon=True,
                             name="memory-migration").start()
        emit("load", "linking.video.feeds", 62)
        cams = discover_cameras(probe=True)
        for c in cams:
            steps.append(c)
            emit("line", f"feed_{c.index} [{'x' if c.available else ' '}] {c.name}", 0)
        self.sig.cameras.emit(cams)
        emit("load", "linking.observer", 85)
        prov = self.observer.provider
        emit("line", f"[observer] {prov + ': ' + self.observer.model_name if prov != 'none' else 'offline: no key'}", 0)
        emit("load", "system.online", 100)
        while time.time() - t0 < (0.5 if quiet else 2.5):
            time.sleep(0.1)
        emit("done", "", 100)

    def _migrate_memory(self, mem, emb, det):
        from ..inference.models import display_name
        """Re-make stored descriptors for a new recognition model from the photos memory keeps (worker thread;
        recognition stays paused meanwhile, tracking runs)."""
        try:
            if mem.migrate_from:
                r = mem.migrate(emb, det)
                self.sig.migrated.emit(f"face memory upgraded: {r['profiles']} profiles, {r['views']} views re-made "
                                       f"from stored photos in {r['seconds']:.0f}s, {r['dropped']} unusable dropped "
                                       f"({display_name(r['from'])} -> {display_name(r['to'])})")
            else:
                r = mem.audit_views(emb, det)
                self.sig.migrated.emit(f"face memory audited: {r['dropped']} of {r['views']} stored views were barely "
                                       f"recognisable and removed ({r['seconds']:.0f}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            mem.migration_progress = None
            self.sig.migrated.emit(f"FAILED: face memory upgrade: {e!r}; recognition paused, restart to retry")

    def on_memory_migrated(self, msg: str):
        failed = msg.startswith("FAILED")
        self.log("MEMORY", msg, "alert" if failed else "")
        self.js(f"G.say({json.dumps('MEMORY UPGRADE FAILED' if failed else 'MEMORY UPGRADED')})")
        self.push_state()

    def on_boot_step(self, kind, text, pct):
        if kind == "line":
            self.js(f"G.bootLine({json.dumps(text)})")
            self.log("BOOT", text)
        elif kind == "load":
            self.js(f"G.bootLoad({json.dumps(text)}, {pct})")
        elif kind == "done":
            if self.args.skip_boot:
                self.on_booted()
            else:
                self.js("G.bootDone()")

    def on_booted(self):
        if self.booted:
            return
        self.booted = True
        self.set_mode(self.mode)
        self.render_timer.start()
        self.state_timer.start()
        self.tick_timer.start()
        if self.args.skip_boot:
            self.js("G.say('WHAT ARE YOUR COMMANDS ?')")
        for spec in self.args.initial_sources:
            if spec.kind == Kind.CAMERA:
                cam = next((c for c in self.cameras if c.index == int(spec.ref)), None)
                if cam:
                    spec.camera, spec.name = cam, cam.name
            self.open_feed(spec)
        if self.args.images:
            self.lab.add_paths(self.args.images)
            self.set_mode("images")

    def on_cameras(self, cams):
        self.cameras = cams

    def open_feed(self, spec: SourceSpec) -> Feed:
        feed = Feed(spec)
        self.hub.add_feed(feed)
        feed.start()
        v = FeedView(feed, self.layers, self.hub)
        v.selected.connect(self.select_feed)
        v.subjectClicked.connect(lambda fid, sid: self.focus(fid, sid))
        v.menuAction.connect(self.on_feed_menu)
        self.grid.add(v)
        v.expandToggled.disconnect()
        v.expandToggled.connect(self.cycle_expand)
        self.select_feed(feed.id)
        self.log("LINK", f"{feed.name} ← {spec.label}")
        if self.mode == "live":
            self.grid.show()
            self.grid.raise_()
        return feed

    def close_feed(self, fid: int):
        f = self.hub.feeds.get(fid)
        if not f:
            return
        if self.grid.expanded == fid:
            self.set_feed_full(None, False)
        self.grid.remove(fid)
        threading.Thread(target=self.hub.remove_feed, args=(fid,), daemon=True).start()
        if self.hub.focus and self.hub.focus[0] == fid:
            self.hub.focus = None
        self.log("UNLINK", f.name)
        if self.selected == fid:
            self.selected = next(iter(k for k in self.hub.feeds if k != fid), None)
            self.grid.set_selected(self.selected)

    def select_feed(self, fid):
        self.selected = fid
        self.grid.set_selected(fid)

    def focus(self, fid, sid):
        if sid is None or (self.hub.focus == (fid, sid)):
            self.hub.focus = None
        else:
            self.hub.focus = (fid, sid)
            self.select_feed(fid)

    def on_feed_menu(self, fid, key):
        f = self.hub.feeds.get(fid)
        if not f:
            return
        if key.startswith("focus:"):
            self.focus(fid, int(key.split(":")[1]))
        elif key.startswith("suppress:"):
            sid = int(key.split(":")[1])
            f.suppress_subject(sid)
            self.log("EXCLUDE", f"{f.name}/S{sid:02d}")
        else:
            self.on_action({"name": key, "args": {"feed": str(fid)}})

    def on_action(self, msg: dict):
        name, a = msg.get("name"), msg.get("args", {})
        fid = int(a["feed"]) if a.get("feed", "").isdigit() else self.selected
        f = self.hub.feeds.get(fid) if fid is not None else None
        if name == "mode":
            self.set_mode(a.get("mode", "live"))
        elif name == "select" and f:
            self.select_feed(fid)
        elif name == "open_camera":
            idx = int(a["index"])
            cam = next((c for c in self.cameras if c.index == idx), None)
            self.open_feed(SourceSpec(Kind.CAMERA, str(idx), cam.name if cam else f"Camera {idx}", camera=cam))
        elif name == "rescan":
            busy = {int(f.spec.ref) for f in self.hub.feeds.values() if f.spec.kind == Kind.CAMERA}
            threading.Thread(target=lambda: self.sig.cameras.emit(discover_cameras(True, skip=busy)), daemon=True).start()
            self.log("SCAN", "enumerating video devices")
        elif name == "add_stream":
            url, ok = QInputDialog.getText(self, "NETWORK FEED", "RTSP / HTTP / UDP URL:")
            if ok and url.strip():
                self.open_feed(SourceSpec(Kind.STREAM, url.strip(), _short_url(url.strip())))
        elif name == "add_file":
            paths, _ = QFileDialog.getOpenFileNames(self, "VIDEO FILES", "", "Video (*.mp4 *.avi *.mkv *.mov *.wmv *.webm *.m4v);;All (*)")
            for p in paths:
                self.open_feed(SourceSpec(Kind.FILE, p))
        elif name == "tracking" and f:
            (f.stop_tracking if f.tracking else f.start_tracking)()
        elif name == "pause" and f:
            f.capture.set_paused(not f.capture.paused)
        elif name == "small" and f:
            f.settings.small_faces = not f.settings.small_faces
        elif name == "landmarks" and f:
            f.settings.landmarks = not f.settings.landmarks
        elif name == "expand" and f:
            self.set_feed_full(None, False)
            self.grid.toggle_expand(fid)
        elif name == "fullscreen" and f:
            self.set_feed_full(fid, not (self.feed_full and self.grid.expanded == fid))
        elif name == "close" and f:
            self.close_feed(fid)
        elif name == "focus":
            sid = a.get("subject")
            self.focus(fid, int(sid) if sid and sid.isdigit() else None)
        elif name == "suppress" and f and a.get("subject", "").isdigit():
            f.suppress_subject(int(a["subject"]))
            self.hub.focus = None
        elif name == "layer":
            self.layers.toggle(a.get("layer", ""))
            if a.get("layer") in Hub.MESH_LAYERS and getattr(self.layers, a.get("layer"), False):
                for x in self.hub.feeds.values():
                    x.settings.landmarks = True
        elif name == "style":
            self.layers.style = a.get("style", "samaritan")
        elif name == "open_images":
            paths, _ = QFileDialog.getOpenFileNames(self, "PHOTOGRAPHS", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
            if paths:
                self.lab.add_paths(paths)
        elif name == "open_folder":
            d = QFileDialog.getExistingDirectory(self, "IMAGE COLLECTION")
            if d:
                self.lab.add_paths([d])
        elif name == "clear_images":
            self.lab.clear()
        elif name == "select_image":
            self.lab.current = int(a.get("index", 0))
            self.lab.inspected = None
            self.on_lab_changed()
        elif name == "inspect":
            self.lab.inspect(a.get("key", ""))
        elif name == "set_ab":
            self.lab.set_ab(a.get("key", ""))
        elif name == "snapshot" and f:
            path = self.snapshot(fid)
            if path:
                self.js("G.say('SNAPSHOT SAVED')")
        elif name == "clear_compare":
            self.lab.ab.clear()
            self.on_lab_changed()
        elif name.startswith("mem_") or name == "remember":
            self._memory_action(name, a)
        self.push_state()

    def _memory_action(self, name: str, a: dict):
        mem = self.hub.memory
        if mem is None:
            return
        pid = int(a["person"]) if str(a.get("person", "")).isdigit() else self.memory_panel.selected
        if name == "mem_select":
            self.memory_panel.selected = pid
        elif name in ("mem_open", "mem_page"):
            if pid is not None and mem.person(pid) is not None:
                self.memory_panel.selected = self.memory_panel.page = pid
                if self.mode != "memory":
                    self.set_mode("memory")
        elif name == "mem_back":
            self.memory_panel.page = None
        elif name == "mem_notes" and pid is not None:
            p = mem.person(pid)
            text, ok = QInputDialog.getMultiLineText(self, "OPERATOR NOTES", f"Notes on {p.name if p else pid}:",
                                                     p.notes if p else "")
            if ok:
                mem.set_notes(pid, text)
                self.log("MEMORY", f"P-{pid:04d} notes updated")
        elif name == "mem_alert" and pid is not None:
            on = pid not in mem.alert_ids()
            mem.set_alert(pid, on)
            self.log("MEMORY", f"P-{pid:04d} " + ("added to the watchlist: alert when seen" if on else
                                                  "removed from the watchlist"))
            self.js(f"G.say({json.dumps('WATCHLIST UPDATED')})")
        elif name == "mem_locate" and pid is not None:
            for f in self.hub.feeds.values():
                r = f.latest()
                t = next((t for t in (r.tracks if r else []) if t.person_id == pid and t.visible), None)
                if t is not None:
                    self.set_mode("live")
                    self.select_feed(f.id)
                    self.focus(f.id, t.id)
                    break
            else:
                self.js("G.say('SUBJECT NOT IN VIEW')")
        elif name in ("cand_confirm", "cand_reject") and str(a.get("cand", "")).isdigit():
            r = mem.resolve_candidate(int(a["cand"]), confirm=name == "cand_confirm")
            self.log("MEMORY", f"possible sighting {r}" + (f" (P-{pid:04d})" if pid is not None else ""))
        elif name == "mem_toggle":
            if mem.migrate_from or mem.needs_audit:
                self.js("G.say('MEMORY UPGRADE IN PROGRESS')")
            else:
                mem.enabled = not mem.enabled
                self.log("MEMORY", "recognition " + ("enabled" if mem.enabled else "paused"))
        elif name == "mem_rename" and pid is not None:
            p = mem.person(pid)
            text, ok = QInputDialog.getText(self, "LABEL PROFILE", f"Label for {p.code if p else pid}:",
                                            text=p.label if p else "")
            if ok:
                mem.rename(pid, text)
                self.log("MEMORY", f"P-{pid:04d} labelled {text!r}")
        elif name == "mem_merge" and pid is not None:
            others = [p for p in mem.persons() if p.id != pid]
            if not others:
                return
            items = [f"{p.code}  {p.label}".strip() for p in others]
            choice, ok = QInputDialog.getItem(self, "MERGE PROFILE",
                                              f"P-{pid:04d} is the same person as:", items, 0, False)
            if ok:
                target = others[items.index(choice)].id
                mem.merge(target, pid)
                self.memory_panel.selected = target
                if self.memory_panel.page is not None:
                    self.memory_panel.page = target
                self.log("MEMORY", f"P-{pid:04d} merged into P-{target:04d}")
        elif name == "mem_merge_into" and pid is not None and str(a.get("target", "")).isdigit():
            target = int(a["target"])
            keep, drop = min(pid, target), max(pid, target)
            if QMessageBox.question(self, "MERGE", f"P-{drop:04d} and P-{keep:04d} are the same person?\n"
                                                    f"P-{drop:04d} will be folded into P-{keep:04d}.") \
                    == QMessageBox.StandardButton.Yes:
                mem.merge(keep, drop)
                self.memory_panel.selected = keep
                if self.memory_panel.page is not None:
                    self.memory_panel.page = keep
                self.log("MEMORY", f"P-{drop:04d} merged into P-{keep:04d}")
        elif name == "mem_forget" and pid is not None:
            if QMessageBox.question(self, "FORGET", f"Permanently delete P-{pid:04d} and its history?") \
                    == QMessageBox.StandardButton.Yes:
                mem.forget(pid)
                self.memory_panel.selected = None
                self.log("MEMORY", f"P-{pid:04d} forgotten")
        elif name == "mem_dismiss" and pid is not None and str(a.get("target", "")).isdigit():
            mem.dismiss_pair(pid, int(a["target"]))
            self.log("MEMORY", f"P-{pid:04d} and P-{int(a['target']):04d} marked as different people")
        elif name == "mem_view_remove" and str(a.get("view", "")).isdigit():
            msg = "Remove this stored view from the profile?\n(Use it when the photo shows someone else.)"
            if QMessageBox.question(self, "REMOVE VIEW", msg) == QMessageBox.StandardButton.Yes:
                ok = mem.remove_view(int(a["view"]))
                self.log("MEMORY", "view removed" if ok else "a profile keeps at least one view")
        elif name == "mem_merge_certain":
            n = sum(1 for *_, sc in mem.duplicate_pairs(threshold=mem.certain_duplicate, limit=10_000))
            msg = (f"{n} profile pair(s) are near-identical (similarity >= {mem.certain_duplicate:.2f}; different "
                   "people never score this high). Fold each into the older profile?\nThis cannot be undone.")
            if n and QMessageBox.question(self, "MERGE CERTAIN DUPLICATES", msg) == QMessageBox.StandardButton.Yes:
                done = mem.merge_certain_duplicates()
                self.log("MEMORY", f"merged {len(done)} certain duplicate profile(s)")
                self.memory_panel.selected = None
        elif name == "mem_review":
            self.memory_panel.review = not self.memory_panel.review
        elif name == "mem_prune":
            ids = mem.prunable(7)
            if not ids:
                self.js("G.say('NOTHING TO PRUNE')")
            elif QMessageBox.question(self, "PRUNE", f"Forget {len(ids)} unlabelled one-off profile(s) "
                                                     "not seen for 7+ days?") == QMessageBox.StandardButton.Yes:
                self.log("MEMORY", f"pruned {mem.prune(7)} one-off profiles")
        elif name == "mem_clear":
            if QMessageBox.question(self, "CLEAR MEMORY", "Permanently delete every profile and all history?") \
                    == QMessageBox.StandardButton.Yes:
                mem.clear()
                self.memory_panel.selected = None
                self.log("MEMORY", "all profiles deleted")
        elif name == "remember":
            r = self.lab.remember(a.get("key", ""))
            if r:
                self.log("MEMORY", r)
                self.js(f"G.say({json.dumps(r.split(' (')[0].upper())})")

    def set_mode(self, mode: str):
        if mode != "live":
            self.feed_full = False
        self._mem_mode_changed = True
        self.mode = mode
        self.grid.setVisible(mode == "live")
        self.imageview.setVisible(mode == "images")
        x, y, w, h = self._hole
        for wdg in (self.grid, self.imageview):
            wdg.setGeometry(x, y, w, h)
            wdg.raise_()
        self.push_state()

    def on_lab_changed(self):
        self.imageview.update()
        if self.mode == "images":
            self.push_state()

    def on_command(self, text: str):
        from ..assistant.keys import mask
        self.log("OPERATOR", mask(text), "op")
        if text.lower().startswith(":model"):
            from ..assistant import keys as _keys
            from ..assistant import openrouter as _orr
            want = text[6:].strip()
            if want:
                _orr.MODEL = want
                _keys.save("or_model", want)
                self.observer.reset()
            self.log("OBSERVER", f"model {_orr.MODEL}")
            self.js(f"G.say({json.dumps('MODEL ' + _orr.MODEL.split('/')[-1].upper())})")
            return
        if text.lower().startswith(":recogniser") or text.lower().startswith(":recognizer"):
            from ..assistant import keys as _keys
            from ..inference.models import recogniser_profile
            want = text.split(None, 1)[1].strip().lower() if len(text.split()) > 1 else ""
            want = {"fused": "full", "arcface": "lite"}.get(want, want)
            if want in ("full", "lite"):
                _keys.save("recogniser", want)
                self.log("MEMORY", f"recogniser set to GothamFaces {want}: applies at next start (face memory is "
                                   "converted automatically from its stored photos)")
                self.js("G.say('RESTART TO APPLY')")
            else:
                self.log("MEMORY", f"recogniser {recogniser_profile().display} · options: :recogniser full "
                                   "(strongest, default) | :recogniser lite (faster in dense crowds)")
            return
        if text.lower().startswith(":key "):
            which = self.observer.set_api_key(text[5:].strip())
            self.log("OBSERVER", f"{which} key stored (encrypted, this PC only) · model {self.observer.model_name}")
            self.js("G.say('KEY ACCEPTED')")
            return
        if self.observer.busy:
            self.js("G.say('PROCESSING PREVIOUS COMMAND')")
            return
        if self._local_command(text):
            return
        self.js("G.busy(true)")
        self.observer.ask(text)

    def _local_command(self, text: str) -> bool:
        t = text.strip().lower()
        if not t.startswith(":"):
            return False
        m = re.match(r":(\w+)\s*(on|off)?$", t)
        if not m:
            return False
        word, onoff = m.groups()
        if word == "snap":
            path = self.snapshot(self.selected)
            self.js(f"G.say({json.dumps('SNAPSHOT SAVED' if path else 'NO FEED SELECTED')})")
            return True
        if word in LAYER_HELP:
            setattr(self.layers, word, onoff != "off" if onoff else not getattr(self.layers, word))
            if getattr(self.layers, word) and word in Hub.MESH_LAYERS:
                for x in self.hub.feeds.values():
                    x.settings.landmarks = True
            self.js(f"G.say({json.dumps(word.upper() + (' ENABLED' if getattr(self.layers, word) else ' DISABLED'))})")
        elif word in ("samaritan", "machine"):
            self.layers.style = word
            self.js(f"G.say({json.dumps(word.upper() + ' MODE')})")
        elif word == "track":
            self.hub.set_tracking(None, onoff != "off")
            self.js("G.say('TRACKING " + ("DISENGAGED" if onoff == "off" else "ENGAGED") + "')")
        elif word == "reset":
            self.observer.reset()
            self.js("G.say('CONTEXT CLEARED')")
        else:
            return False
        return True

    def on_reply(self, head: str, body: str):
        self.js("G.busy(false); G.status('')")
        self.js(f"G.say({json.dumps(head)})")
        self.log("SAMARITAN", head, "ai")
        for ln in body.splitlines():
            if ln.strip():
                self.log("·", ln.strip(), "ai")

    def on_observer_error(self, msg: str):
        self.js("G.busy(false); G.status('')")
        if msg == "busy":
            return
        self.js(f"G.say({json.dumps(msg.split(' — ')[0])})")
        self.log("FAULT", msg, "alert")

    def on_render(self):
        if self.mode == "live":
            self.grid.refresh()

    def on_tick(self):
        self.hub.tick()
        events, alerts = self.hub.drain()
        for e in events:
            self.log(e.kind.upper(), f"FEED-{e.feed_id:02d}" + (f"/S{e.subject:02d}" if e.subject else "") + f" {e.text}",
                     "alert" if e.kind in ("signal_lost",) else "")
        for a in alerts:
            self.log("ALERT", f"{a.title}: {a.text}", "alert")
            self.js(f"G.alert({json.dumps(a.title)}, {json.dumps(a.text)})")
            if self.tray is not None and not self.isActiveWindow():
                self.tray.showMessage(f"GOTHAM · {a.title}", a.text, QSystemTrayIcon.MessageIcon.Warning, 6000)
        self._announce(events)
        while self.hub.ui_actions:
            act = self.hub.ui_actions.popleft()
            if act.get("action") == "focus":
                self.select_feed(act["feed"])
            elif act.get("action") == "snapshot":
                path = self.snapshot(act.get("feed") or self.selected)
                act["result"] = path

    def _announce(self, events):
        now = time.monotonic()
        for e in events:
            line = None
            if e.kind == "recognized" and e.person:
                info = self.hub.person_info(e.person)
                if info and info["label"] and now - self._said.get(("p", e.person), -1e9) > 300:
                    line, key = f"{info['label']} IDENTIFIED", ("p", e.person)
            elif e.kind == "transit" and e.person:
                if now - self._said.get(("t", e.person), -1e9) > 120:
                    line, key = f"{self.hub.person_name(e.person)} RELOCATED FEED {e.feed_id:02d}", ("t", e.person)
            elif e.kind == "new_person":
                if now - self._said.get("new", -1e9) > 30:
                    line, key = "NEW SUBJECT ENROLLED", "new"
            if line and now - self._last_say > 4:
                self._said[key] = now
                self._last_say = now
                self.js(f"G.say({json.dumps(line.upper())})")

    def snapshot(self, fid: int | None) -> str | None:
        v = next((v for v in self.grid.views if v.feed.id == fid), None)
        if v is None:
            return None
        d = Path.home() / "Pictures" / "Gotham"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{v.feed.name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        v.grab().save(str(path))
        self.log("SNAPSHOT", str(path))
        return str(path)

    def push_state(self):
        if not self._page_ready:
            return
        now = time.monotonic()
        feeds = []
        for f in self.hub.feeds.values():
            st = f.stats
            r = f.latest()
            feeds.append({
                "id": f.id, "name": f.name, "label": f.spec.label.upper()[:28], "kind": f.spec.kind.value,
                "status": st.status.value, "fps": f"{st.fps:.0f}", "afps": f"{r.process_fps:.0f}" if r else "0",
                "lat": f"{r.latency_ms:.0f}" if r else "--", "tracking": f.tracking, "paused": f.capture.paused,
                "small": f.settings.small_faces, "landmarks": f.settings.landmarks, "reconnects": st.reconnects,
                "dropped": st.dropped, "res": f"{st.width}x{st.height}" if st.width else "",
                "session_s": now - f.tracker.started_at if f.tracking and f.tracker.started_at else 0,
                "total": f.tracker.total_subjects, "faces": len(r.faces) if r else 0,
                "spark": [n for _, n in list(self.hub.activity.get(f.id, ()))[-300:]][::5],
                "faults": f.faults})
        open_cams = {int(f.spec.ref) for f in self.hub.feeds.values() if f.spec.kind == Kind.CAMERA}
        state = {
            "mode": self.mode, "selected": self.selected, "feeds": feeds,
            "sources": [{"index": c.index, "name": c.name, "w": c.width, "h": c.height, "available": c.available,
                         "open": c.index in open_cams} for c in self.cameras],
            "layers": self.layers.as_dict(), "focus": list(self.hub.focus) if self.hub.focus else None,
            "system": self._system(),
        }
        state["memory_summary"] = self.memory_panel.summary()
        if self.mode == "live":
            state["subjects"], state["inspector"] = self._subjects()
        elif self.mode == "memory":
            mp = self.memory_panel.payload(self.hub._live_people())
            if mp is not getattr(self, "_mem_sent", None) or getattr(self, "_mem_mode_changed", True):
                state["memory"] = mp
                self._mem_sent = mp
                self._mem_mode_changed = False
        else:
            state["images"], state["lab"] = self.lab.payload()
        self.js(f"G.state({json.dumps(state, default=str)})")
        if self._log_buf:
            self.js(f"G.log({json.dumps(self._log_buf)})")
            self._log_buf = []

    def _subjects(self):
        f = self.hub.feeds.get(self.selected) if self.selected else None
        if not f:
            return [], None
        r = f.latest()
        if not r:
            return [], None
        now = time.monotonic()
        subs, insp = [], None
        tracks = [t for t in r.tracks if not t.suppressed and
                  (t.state != "lost" or now - t.last_seen < 10 or self.hub.focus == (f.id, t.id))]
        for t in sorted(tracks, key=lambda t: (not t.visible, t.state == "lost", -t.last_seen))[:12]:
            if t.suppressed:
                continue
            g = t.face.gaze
            gaze = None
            if g is not None and t.visible:
                gaze = "CAMERA" if g.reliable and abs(g.yaw) < 12 and abs(g.pitch) < 12 else f"{g.yaw:+.0f}°/{g.pitch:+.0f}°"
            subs.append({"id": t.id, "label": t.label, "state": "occluded" if t.state == "coasting" else t.state,
                         "visible": t.visible, "age": now - t.first_seen, "reacq": t.reacquired, "gaze": gaze,
                         "quality": t.face.quality.get("overall", 0), "person": self.hub.person_name(t.person_id),
                         "pid": t.person_id, "mem": t.memory_state, "pscore": f"{t.person_score:.2f}"})
            if self.hub.focus == (f.id, t.id):
                fc = t.face
                insp = {"id": t.id, "feed": f.id, "label": t.label, "state": "occluded" if t.state == "coasting" else t.state,
                        "person": self.hub.person_name(t.person_id), "pid": t.person_id, "mem": t.memory_state,
                        "pscore": f"{t.person_score:.2f}",
                        "size": int(fc.box[3]), "score": f"{fc.score:.2f}", "trunc": fc.truncated,
                        "quality": {k: v for k, v in fc.quality.items() if k != "detection"}}
                if fc.pose:
                    insp["pose"] = [f"{fc.pose.yaw:+.0f}", f"{fc.pose.pitch:+.0f}", f"{fc.pose.roll:+.0f}"]
                if g:
                    insp["gaze"] = gaze or "--"
                    insp["at_cam"] = gaze == "CAMERA"
                if fc.eye_open:
                    insp["eyes"] = f"R {fc.eye_open[0]:.2f} · L {fc.eye_open[1]:.2f}"
                if fc.blend:
                    top = sorted(((k, v) for k, v in fc.blend.items() if k != "_neutral"), key=lambda kv: -kv[1])[:2]
                    insp["expr"] = " · ".join(f"{k[:14]} {v:.2f}" for k, v in top)
        return subs, insp

    def _system(self) -> dict:
        prov = next((f.provider for f in self.hub.feeds.values() if f.provider), "")
        up = int(time.time() - self.hub.started)
        return {"cpu": f"{psutil.cpu_percent(None):.0f}", "ram": f"{psutil.virtual_memory().percent:.0f}",
                "proc": f"{self._proc.memory_info().rss / 2**20:.0f}",
                "provider": "DIRECTML" if "Dml" in prov else ("CPU" if prov else "IDLE"),
                "gpu": self.args.gpu_name,
                "observer": ("BUSY" if self.observer.busy else "READY" if self.observer.provider != "none" else "NO KEY")
                + (f" · ${self.observer.spent:.4f}" if self.observer.spent else ""),
                "watches": sum(1 for w in self.hub.watches.values() if w.active),
                "uptime": f"{up // 3600:02d}:{up % 3600 // 60:02d}:{up % 60:02d}"}

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, e):
        self.render_timer.stop()
        self.state_timer.stop()
        self.tick_timer.stop()
        for fid in list(self.hub.feeds):
            self.hub.feeds[fid].stop()
        if self.hub.memory is not None:
            self.hub.memory.close()
        super().closeEvent(e)


def _short_url(u: str) -> str:
    u = re.sub(r"//[^@/]*@", "//", u)
    return u.split("://", 1)[-1][:40]
