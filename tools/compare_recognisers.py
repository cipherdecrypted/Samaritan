from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pairs(base: Path):
    lines = (base / "pairsDevTest.txt").read_text().splitlines()
    pairs = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) == 3:
            pairs.append(((p[0], int(p[1])), (p[0], int(p[2])), True))
        elif len(p) == 4:
            pairs.append(((p[0], int(p[1])), (p[2], int(p[3])), False))
    return pairs


def degrade(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    if kind == "tiny":
        s = 26 / (h / 3)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 2.2)
    if kind == "dark_noise":
        x = img.astype(np.float32) * 0.22 + rng.normal(0, 6, img.shape)
        return np.clip(x, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 8])[1], 1)
    if kind == "combo":
        x = degrade(img, "tiny", rng)
        x = np.clip(x.astype(np.float32) * 0.45 + rng.normal(0, 4, x.shape), 0, 255).astype(np.uint8)
        return degrade(x, "jpeg", rng)
    raise ValueError(kind)


from gotham.inference.embedder import align112
from gotham.inference.models import make_detector
from gotham.inference.runtime import Session
from gotham.perf import tune_process

MODELS = {
    "ArcFace R50": ROOT / "models" / "w600k_r50_batch.onnx",
    "ArcFace R50 fp16": ROOT / "models" / "w600k_r50_batch_fp16.onnx",
    "AdaFace IR101": ROOT / "models" / "adaface_ir101_webface12m.onnx",
    "AdaFace IR101 fp16": ROOT / "models" / "adaface_ir101_webface12m_fp16.onnx",
}


def shrink(img, face_px):
    h, w = img.shape[:2]
    s = face_px / (h / 3)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CONDITIONS = {
    "clean": lambda im, rng: im,
    "blur": lambda im, rng: degrade(im, "blur", rng),
    "dark+noise": lambda im, rng: degrade(im, "dark_noise", rng),
    "jpeg q8": lambda im, rng: degrade(im, "jpeg", rng),
    "face 26px": lambda im, rng: degrade(im, "tiny", rng),
    "face 20px": lambda im, rng: shrink(im, 20),
    "face 16px": lambda im, rng: shrink(im, 16),
    "cctv combo": lambda im, rng: degrade(im, "combo", rng),
}
HARD = {
    "12px*": lambda im, rng: shrink(im, 12),
    "16px+jpeg*": lambda im, rng: degrade(shrink(im, 16), "jpeg", rng),
    "20px+dark+jpeg*": lambda im, rng: degrade(degrade(shrink(im, 20), "dark_noise", rng), "jpeg", rng),
    "cctv combo*": lambda im, rng: degrade(im, "combo", rng),
}


def blob(crops):
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32)
    return ((x - 127.5) / 127.5).transpose(0, 3, 1, 2).copy()


def embed(sess: Session, crops: list[np.ndarray], flip: bool) -> np.ndarray:
    out = []
    for i in range(0, len(crops), 8):
        chunk = crops[i:i + 8]
        b = blob(chunk)
        pad = 8 - len(chunk)
        if pad:
            b = np.concatenate([b, np.zeros((pad,) + b.shape[1:], np.float32)])
        f = sess.run(b)[0][:len(chunk)]
        if flip:
            f = f + sess.run(b[..., ::-1].copy())[0][:len(chunk)]
        out.append(f)
    v = np.concatenate(out).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def metrics(gen: np.ndarray, imp: np.ndarray):
    imp_sorted = np.sort(imp)
    out = []
    for far in (1e-3, 1e-4):
        t = imp_sorted[int(np.ceil(len(imp) * (1 - far))) - 1]
        out.append(float(np.mean(gen > t)))
    ths = np.linspace(-0.2, 1.0, 1201)
    frr = np.array([np.mean(gen < t) for t in ths])
    far_ = 1 - np.searchsorted(imp_sorted, ths) / len(imp)
    i = int(np.argmin(np.abs(frr - far_)))
    return out[0], out[1], float((frr[i] + far_[i]) / 2), float(np.median(gen))


def main(base: Path, flip: bool):
    tune_process()
    pairs = load_pairs(base)
    det = make_detector()

    def kps(img):
        ds = det.detect(img, threshold=0.3)
        if not ds:
            return None
        c = np.array([img.shape[1] / 2, img.shape[0] / 2])
        return min(ds, key=lambda d: np.linalg.norm(d.box[:2] + d.box[2:] / 2 - c)).kps

    def load(name, idx):
        return cv2.imread(str(base / "lfw" / name / f"{name}_{idx:04d}.jpg"))

    t0 = time.time()
    gal, gal_names = [], []
    for (na, ia), _, _ in pairs:
        im = load(na, ia)
        k = kps(im)
        gal.append(align112(im, k) if k is not None else None)
        gal_names.append(na)
    probes: dict[str, list] = {}
    misses = {}
    clean_k = {}
    conds = HARD if "--hard" in sys.argv else {**CONDITIONS, **HARD}
    for cond, fn in conds.items():
        crops, miss = [], 0
        for i, (_, (nb, ib), _) in enumerate(pairs):
            raw = load(nb, ib)
            im = fn(raw, np.random.default_rng(i))
            if cond in HARD:
                if i not in clean_k:
                    clean_k[i] = kps(raw)
                k = clean_k[i]
            else:
                k = kps(im)
            if k is None:
                miss += 1
            crops.append(align112(im, k) if k is not None else None)
        probes[cond], misses[cond] = crops, miss
    probe_names = [nb for _, (nb, _), _ in pairs]
    same = np.array([s for *_, s in pairs])
    print(f"faces prepared in {time.time() - t0:.0f}s; probes not detected: {misses}")

    ok_g = np.array([g is not None for g in gal])
    gn = np.array(gal_names)
    print(f"{'model':22s} {'condition':11s} {'TAR@1e-3':>8s} {'TAR@1e-4':>8s} {'EER':>6s} {'gen med':>7s}")
    scores: dict[str, dict[str, np.ndarray]] = {}

    def report(mname, cond, S, ok_p):
        idx = np.arange(len(pairs))
        gmask = same & ok_g & ok_p
        gen = S[idx[gmask], idx[gmask]]
        diff = gn[:, None] != np.array(probe_names)[None, :]
        imp = S[np.ix_(ok_g, ok_p)][diff[np.ix_(ok_g, ok_p)]]
        t3, t4, eer, med = metrics(gen, imp)
        print(f"{mname:22s} {cond:11s} {t3:8.3f} {t4:8.3f} {eer:6.3f} {med:7.3f}", flush=True)

    only = [m for m in MODELS if "--fp32" not in sys.argv or "fp16" not in m]
    for mname in only:
        path = MODELS[mname]
        if not path.exists():
            print(f"{mname}: {path.name} missing")
            continue
        sess = Session(path)
        G = np.zeros((len(gal), 512), np.float32)
        G[ok_g] = embed(sess, [g for g in gal if g is not None], flip)
        for cond, crops in probes.items():
            ok_p = np.array([c is not None for c in crops])
            P = np.zeros((len(crops), 512), np.float32)
            P[ok_p] = embed(sess, [c for c in crops if c is not None], flip)
            S = G @ P.T
            scores.setdefault(mname, {})[cond] = S
            report(mname, cond, S, ok_p)
    a, b = "ArcFace R50", "AdaFace IR101"
    if a in scores and b in scores:
        for w in (0.5, 0.6):
            for cond, crops in probes.items():
                ok_p = np.array([c is not None for c in crops])
                report(f"FUSION {1 - w:.1f}A+{w:.1f}D", cond, (1 - w) * scores[a][cond] + w * scores[b][cond], ok_p)


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--flip" in sys.argv)
