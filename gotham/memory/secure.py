from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

_ENTROPY = b"gotham.memory.v1"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buf = ctypes.create_string_buffer(data, len(data))
    b = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    b._keep = buf
    return b


AVAILABLE = sys.platform == "win32"
if AVAILABLE:
    _crypt32 = ctypes.WinDLL("crypt32")
    _kernel32 = ctypes.WinDLL("kernel32")
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p


def _call(fn, data: bytes) -> bytes:
    src, ent, out = _blob(data), _blob(_ENTROPY), _Blob()
    if not fn(ctypes.byref(src), None, ctypes.byref(ent), None, None, 0x1, ctypes.byref(out)):
        raise OSError(ctypes.get_last_error() or "DPAPI failure")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        _kernel32.LocalFree(out.pbData)


def protect(data: bytes) -> bytes:
    if not AVAILABLE:
        return data
    return _call(lambda s, d, e, r, p, f, o: _crypt32.CryptProtectData(s, d, e, r, p, f, o), data)


def unprotect(data: bytes) -> bytes:
    if not AVAILABLE:
        return data
    return _call(lambda s, d, e, r, p, f, o: _crypt32.CryptUnprotectData(s, None, e, r, p, f, o), data)
