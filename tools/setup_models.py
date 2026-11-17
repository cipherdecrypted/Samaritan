from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOWNLOADS = [
    ("face_detection_yunet_2023mar.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    ("face_recognition_sface_2021dec.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    ("face_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
]
BUNDLE = ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
          "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
          {"det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
           "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"})
ADAFACE_WEIGHTS = ("https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/"
                   "54f602a0737bd1ee4a4e7e9fd089a485f397fefd/model.safetensors",
                   "2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56")
BUILT_ARCFACE = ["w600k_r50_batch.onnx", "w600k_r50_batch_fp16.onnx"]
BUILT_ADAFACE = ["adaface_ir101_webface12m.onnx", "adaface_ir101_webface12m_fp16.onnx"]
BUILD_PACKAGES = ["numpy", "safetensors", "onnx", "onnxscript", "onnxconverter-common", "onnxruntime"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, digest: str):
    if dest.exists() and sha256(dest) == digest:
        print(f"  ok      {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gotham-setup"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct >= 0 and pct // 10 != shown:
                shown = pct // 10
                print(f"\r  fetch   {dest.name} {pct:3d}%", end="", flush=True)
    print()
    if sha256(tmp) != digest:
        tmp.unlink()
        raise SystemExit(f"checksum mismatch for {dest.name}: the download is not the expected file")
    tmp.replace(dest)


def build_env(where: Path) -> Path:
    py = where / "Scripts" / "python.exe"
    if not py.exists():
        print("  creating build environment (one-time, removed afterwards)")
        subprocess.run([sys.executable, "-m", "venv", str(where)], check=True)
        pip = [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        subprocess.run(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        subprocess.run(pip + BUILD_PACKAGES, check=True)
    return py


def main():
    ap = argparse.ArgumentParser(description="Download and build every model Gotham needs.")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--keep-build-env", action="store_true")
    args = ap.parse_args()
    models = args.models.resolve()
    models.mkdir(parents=True, exist_ok=True)
    cache = models / ".download"
    cache.mkdir(exist_ok=True)

    print("[1/3] downloading models")
    for name, url, digest in DOWNLOADS:
        download(url, models / name, digest)
    url, digest, members = BUNDLE
    if not all((models / m).exists() and sha256(models / m) == h for m, h in members.items()):
        bundle = cache / "buffalo_l.zip"
        download(url, bundle, digest)
        with zipfile.ZipFile(bundle) as z:
            for member, h in members.items():
                src = next(n for n in z.namelist() if n.endswith(member))
                with z.open(src) as fi, open(models / member, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                if sha256(models / member) != h:
                    raise SystemExit(f"checksum mismatch for {member}")
                print(f"  ok      {member}")
    else:
        for m in members:
            print(f"  ok      {m}")

    need_arc = not all((models / f).exists() for f in BUILT_ARCFACE)
    need_ada = not all((models / f).exists() for f in BUILT_ADAFACE)
    print("[2/3] building GothamFaces model files")
    if need_arc or need_ada:
        build_dir = Path(tempfile.gettempdir()) / "gotham_build"
        quiet = {**os.environ, "PYTHONWARNINGS": "ignore"}
        py = build_env(build_dir)
        if need_arc:
            subprocess.run([str(py), str(ROOT / "tools" / "make_batch_arcface.py"), str(models)], check=True, env=quiet)
        if need_ada:
            weights = cache / "adaface.safetensors"
            download(*ADAFACE_WEIGHTS[:1], weights, ADAFACE_WEIGHTS[1])
            subprocess.run([str(py), str(ROOT / "tools" / "export_adaface.py"), str(weights),
                            str(models / BUILT_ADAFACE[0])], check=True, env=quiet)
        if not args.keep_build_env:
            shutil.rmtree(build_dir, ignore_errors=True)
    else:
        for f in BUILT_ARCFACE + BUILT_ADAFACE:
            print(f"  ok      {f}")
    shutil.rmtree(cache, ignore_errors=True)

    print("[3/3] checking")
    expected = [n for n, _, _ in DOWNLOADS] + list(members) + BUILT_ARCFACE + BUILT_ADAFACE
    missing = [n for n in expected if not (models / n).exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)}")
    size = sum((models / n).stat().st_size for n in expected) / 1e6
    print(f"  all {len(expected)} model files ready in {models} ({size:.0f} MB)")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
