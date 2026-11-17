from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import config


def _gpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.upper()[:26] or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _install_crash_log():
    import datetime
    import faulthandler
    import threading
    import traceback
    from pathlib import Path

    d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Gotham"
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / "gotham.log", "a", buffering=1, encoding="utf-8")
    f.write(f"\n==== start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} pid {os.getpid()}\n")
    faulthandler.enable(f, all_threads=True)

    def hook(et, ev, tb):
        f.write("".join(traceback.format_exception(et, ev, tb)))
    sys.excepthook = hook
    threading.excepthook = lambda a: f.write(f"[thread {a.thread.name}] " +
                                             "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
    if sys.stderr is None:
        sys.stderr = sys.stdout = f
        try:
            os.dup2(f.fileno(), 2)
        except OSError:
            pass
    return f


def main(argv=None):
    log = _install_crash_log()
    from .perf import tune_process
    log.write(f"[perf] {tune_process()}\n")
    ap = argparse.ArgumentParser(prog="gotham")
    ap.add_argument("--camera", type=int, action="append", default=[], help="open camera index on start")
    ap.add_argument("--file", action="append", default=[], help="open a video file on start")
    ap.add_argument("--stream", action="append", default=[], help="open an RTSP/HTTP stream on start")
    ap.add_argument("--images", nargs="*", default=[], help="images or folders for the image lab")
    ap.add_argument("--skip-boot", action="store_true", help="skip the boot sequence")
    ap.add_argument("--windowed", action="store_true", help="start in a borderless window instead of fullscreen")
    ap.add_argument("--script", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    gpu = "" if os.environ.get("GOTHAM_WEB_GPU") == "1" else "--disable-gpu --disable-gpu-compositing "
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS",
                          gpu + "--enable-logging=stderr --log-level=2 --disable-background-timer-throttling "
                          "--disable-renderer-backgrounding --disable-backgrounding-occluded-windows "
                          "--disable-features=CalculateNativeWinOcclusion")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from .sources.capture import Kind, SourceSpec
    from .ui import assets
    from .ui.main_window import MainWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Gotham")
    assets.load_fonts()
    icon = assets.IMG / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    args.initial_sources = ([SourceSpec(Kind.CAMERA, str(i), f"Camera {i}") for i in args.camera]
                            + [SourceSpec(Kind.FILE, p) for p in args.file]
                            + [SourceSpec(Kind.STREAM, u) for u in args.stream])
    args.gpu_name = _gpu_name()
    w = MainWindow(args)
    test_geometry = os.environ.get("GOTHAM_TEST_GEOMETRY")
    if test_geometry:
        w.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        w.setGeometry(*(int(v) for v in test_geometry.split(",")))
        w.show()
    elif args.windowed:
        w.show()
    else:
        w.showFullScreen()
    if args.script:
        from .ui.selftest import run_script
        run_script(w, args.script)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
