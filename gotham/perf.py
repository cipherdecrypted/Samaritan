from __future__ import annotations

import ctypes
import glob
import os
import sys

BLAS_THREADS = 4

_PROCESS_POWER_THROTTLING = 4
_EXECUTION_SPEED = 0x1
_IGNORE_TIMER_RESOLUTION = 0x4


def disable_power_throttling() -> bool:
    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes

        class _State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        for mask in (_EXECUTION_SPEED | _IGNORE_TIMER_RESOLUTION, _EXECUTION_SPEED):
            st = _State(1, mask, 0)
            if k32.SetProcessInformation(k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                         ctypes.byref(st), ctypes.sizeof(st)):
                return True
        return False
    except Exception:
        return False


def limit_blas_threads(n: int = BLAS_THREADS) -> int | None:
    try:
        import numpy
        libs = os.path.join(os.path.dirname(numpy.__file__), os.pardir, "numpy.libs")
        for path in glob.glob(os.path.join(libs, "*openblas*")):
            lib = ctypes.CDLL(path)
            for setter, getter in (("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                                   ("openblas_set_num_threads", "openblas_get_num_threads")):
                if hasattr(lib, setter):
                    getattr(lib, setter)(int(n))
                    return int(getattr(lib, getter)()) if hasattr(lib, getter) else int(n)
    except Exception:
        pass
    return None


def tune_process() -> dict:
    return {"power_throttling_off": disable_power_throttling(), "blas_threads": limit_blas_threads()}
