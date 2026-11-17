from __future__ import annotations

import json
import os

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def run_script(win, script: str):
    steps = json.loads(open(script, encoding="utf-8").read()) if script.endswith(".json") else json.loads(script)

    def do(action, arg):
        if action == "shot":
            if not os.environ.get("GOTHAM_TEST_GEOMETRY"):
                win.raise_()
                win.activateWindow()
            win.grab().save(arg)
            print("SHOT", arg, flush=True)
        elif action == "action":
            win.on_action(arg)
        elif action == "command":
            win.on_command(arg)
        elif action == "js":
            win.js(arg)
        elif action == "jsq":
            win.page.runJavaScript(arg, 0, lambda r: print("JSQ", r, flush=True))
        elif action == "wait_feed":
            pass
        elif action == "eval":
            print("EVAL", eval(arg, {"w": win}), flush=True)
        elif action == "quit":
            win.close()
            QApplication.quit()

    for t, action, arg in steps:
        QTimer.singleShot(int(t * 1000), lambda a=action, b=arg: do(a, b))
