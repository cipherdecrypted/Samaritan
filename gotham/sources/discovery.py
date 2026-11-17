from __future__ import annotations

import ctypes
import sys
import threading
import uuid
from ctypes import POINTER, WINFUNCTYPE, byref, c_void_p
from dataclasses import dataclass

import cv2

from .. import config


DEVICE_LOCK = threading.RLock()


@dataclass
class CameraInfo:
    index: int
    name: str
    path: str = ""
    width: int = 0
    height: int = 0
    available: bool = True
    msmf_index: int | None = None

    @property
    def backend(self) -> str:
        return "MSMF" if self.msmf_index is not None else "DSHOW"


def open_camera(info: "CameraInfo | int", width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    if isinstance(info, int):
        info = CameraInfo(info, f"Camera {info}")
    with DEVICE_LOCK:
        cap = None
        if info.msmf_index is not None:
            cap = cv2.VideoCapture(info.msmf_index, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(info.index, cv2.CAP_DSHOW)
        if cap.isOpened() and width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap


def release(cap: cv2.VideoCapture):
    with DEVICE_LOCK:
        cap.release()


def _dshow_names() -> list[dict]:
    if sys.platform != "win32":
        return []
    from ctypes import wintypes

    ole32 = ctypes.OleDLL("ole32")
    oleaut32 = ctypes.WinDLL("oleaut32")

    class GUID(ctypes.Structure):
        _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

    def guid(s):
        g = GUID()
        ctypes.memmove(byref(g), uuid.UUID(s).bytes_le, 16)
        return g

    class VARIANT(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                    ("r3", ctypes.c_ushort), ("val", c_void_p), ("val2", c_void_p)]

    def vcall(obj, idx, restype, *argtypes):
        vt = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        return WINFUNCTYPE(restype, c_void_p, *argtypes)(vt[idx])

    CLSID_SystemDeviceEnum = guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86")
    IID_ICreateDevEnum = guid("29840822-5B84-11D0-BD3B-00A0C911CE86")
    CLSID_VideoInputDeviceCategory = guid("860BB310-5D01-11d0-BD3B-00A0C911CE86")
    IID_IPropertyBag = guid("55272A00-42CB-11CE-8135-00AA004BB851")

    out: list[dict] = []
    try:
        ole32.CoInitializeEx(None, 0x2)
    except OSError:
        pass
    dev = c_void_p()
    try:
        ole32.CoCreateInstance(byref(CLSID_SystemDeviceEnum), None, 1, byref(IID_ICreateDevEnum), byref(dev))
    except OSError:
        return out
    en = c_void_p()
    vcall(dev, 3, ctypes.c_long, POINTER(GUID), POINTER(c_void_p), ctypes.c_ulong)(
        dev, byref(CLSID_VideoInputDeviceCategory), byref(en), 0)
    if en.value:
        while True:
            mon = c_void_p()
            fetched = ctypes.c_ulong()
            hr = vcall(en, 3, ctypes.c_long, ctypes.c_ulong, POINTER(c_void_p), POINTER(ctypes.c_ulong))(
                en, 1, byref(mon), byref(fetched))
            if hr != 0 or not fetched.value:
                break
            bag = c_void_p()
            props = {}
            if vcall(mon, 9, ctypes.c_long, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p))(
                    mon, None, None, byref(IID_IPropertyBag), byref(bag)) == 0 and bag.value:
                for key in ("FriendlyName", "DevicePath"):
                    v = VARIANT()
                    r = vcall(bag, 3, ctypes.c_long, wintypes.LPCWSTR, POINTER(VARIANT), c_void_p)(bag, key, byref(v), None)
                    if r == 0 and v.vt == 8:
                        props[key] = ctypes.wstring_at(v.val)
                        oleaut32.VariantClear(byref(v))
                vcall(bag, 2, ctypes.c_ulong)(bag)
            vcall(mon, 2, ctypes.c_ulong)(mon)
            out.append(props)
        vcall(en, 2, ctypes.c_ulong)(en)
    vcall(dev, 2, ctypes.c_ulong)(dev)
    return out


def resolve_camera(index: int) -> CameraInfo:
    names = _dshow_names()
    mf = 0
    for i, meta in enumerate(names):
        if i == index:
            return CameraInfo(i, meta.get("FriendlyName", f"Camera {i}"), meta.get("DevicePath", ""),
                              msmf_index=mf if meta.get("DevicePath") else None)
        if meta.get("DevicePath"):
            mf += 1
    return CameraInfo(index, f"Camera {index}")


def discover_cameras(probe: bool = True, max_index: int = 8, skip: set[int] | None = None) -> list[CameraInfo]:
    names = _dshow_names()
    skip = skip or set()
    count = len(names) if names else max_index
    cams: list[CameraInfo] = []
    mf = 0
    for i in range(count):
        meta = names[i] if i < len(names) else {}
        info = CameraInfo(i, meta.get("FriendlyName", f"Camera {i}"), meta.get("DevicePath", ""))
        if info.path:
            info.msmf_index = mf
            mf += 1
        if probe and i not in skip:
            cap = open_camera(info)
            ok = cap.isOpened()
            if ok:
                ok = False
                for _ in range(10):
                    r, frame = cap.read()
                    if r and frame is not None:
                        info.height, info.width = frame.shape[:2]
                        ok = True
                        break
            release(cap)
            info.available = bool(ok)
            if not names and not ok:
                continue
        cams.append(info)
    return cams
