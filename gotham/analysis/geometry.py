from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

R_EYE = dict(outer=33, inner=133, top=159, bottom=145, iris=468, iris_ring=(469, 470, 471, 472))
L_EYE = dict(outer=263, inner=362, top=386, bottom=374, iris=473, iris_ring=(474, 475, 476, 477))
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10
MOUTH_L, MOUTH_R, LIP_TOP, LIP_BOTTOM = 291, 61, 13, 14
CHEEK_R, CHEEK_L = 234, 454
NOSE_R, NOSE_L = 64, 294
BROW_R, BROW_L = 105, 334

R_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
R_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
L_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]


@dataclass
class Pose:
    yaw: float
    pitch: float
    roll: float
    R: np.ndarray
    source: str


@dataclass
class Gaze:
    yaw: float
    pitch: float
    eye_yaw: float
    eye_pitch: float
    origin_r: np.ndarray
    origin_l: np.ndarray
    reliable: bool


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _euler(R: np.ndarray) -> tuple[float, float, float]:
    ax = R[:, 0]
    f = -R[:, 2]
    yaw = np.degrees(np.arctan2(f[0], -f[2]))
    pitch = np.degrees(np.arctan2(-f[1], np.hypot(f[0], f[2])))
    roll = np.degrees(np.arctan2(ax[1], ax[0]))
    return float(yaw), float(pitch), float(roll)


def pose_from_mesh(P: np.ndarray) -> Pose:
    ax = _unit(P[L_EYE["outer"]] - P[R_EYE["outer"]])
    down = _unit(P[CHIN] - P[FOREHEAD])
    az = _unit(np.cross(ax, down))
    ay = np.cross(az, ax)
    R = np.stack([ax, ay, az], 1)
    return Pose(*_euler(R), R=R, source="mesh")


_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def pose_from_matrix(mat: np.ndarray) -> Pose:
    U, _, Vt = np.linalg.svd(mat[:3, :3].astype(np.float64))
    R = _GL_TO_CV @ (U @ Vt) @ _GL_TO_CV
    return Pose(*_euler(R), R=R, source="mesh")


_MODEL5 = np.array([[-32.0, -35.0, 30.0], [32.0, -35.0, 30.0], [0.0, 0.0, 0.0],
                    [-26.0, 32.0, 25.0], [26.0, 32.0, 25.0]], np.float64)


def pose_from_5pt(kps: np.ndarray, frame_w: int, frame_h: int) -> Pose | None:
    f = float(max(frame_w, frame_h))
    K = np.array([[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL5, kps.astype(np.float64), K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    Rm, _ = cv2.Rodrigues(rvec)
    return Pose(*_euler(Rm), R=Rm, source="5pt")


def eye_openness(P: np.ndarray) -> tuple[float, float]:
    def ear(e):
        w = np.linalg.norm(P[e["outer"], :2] - P[e["inner"], :2])
        h = np.linalg.norm(P[e["top"], :2] - P[e["bottom"], :2])
        return float(h / w) if w > 1e-6 else 0.0
    return ear(R_EYE), ear(L_EYE)


def gaze_from_mesh(P: np.ndarray, pose: Pose, blend: dict[str, float]) -> Gaze:
    hs, vs = [], []
    for e in (R_EYE, L_EYE):
        a, b = P[e["outer"], :2], P[e["inner"], :2]
        lo, hi = (a, b) if a[0] <= b[0] else (b, a)
        axis = hi - lo
        w = np.linalg.norm(axis)
        if w < 1e-6:
            continue
        u = axis / w
        n = np.array([-u[1], u[0]])
        centre = (lo + hi) / 2
        iris = P[e["iris"], :2]
        hs.append(float(np.dot(iris - centre, u) / (w / 2)))
        lid = np.linalg.norm(P[e["top"], :2] - P[e["bottom"], :2])
        vs.append(float(np.dot(iris - (P[e["top"], :2] + P[e["bottom"], :2]) / 2, n) / max(lid, 1e-6)))
    h = float(np.clip(np.mean(hs), -1, 1)) if hs else 0.0
    up = (blend.get("eyeLookUpLeft", 0) + blend.get("eyeLookUpRight", 0)) / 2
    dn = (blend.get("eyeLookDownLeft", 0) + blend.get("eyeLookDownRight", 0)) / 2
    v = up - dn if blend else -float(np.clip(np.mean(vs), -1, 1)) if vs else 0.0
    eye_yaw = h * 38.0
    eye_pitch = v * 28.0
    l_open, r_open = (1 - blend.get("eyeBlinkLeft", 0), 1 - blend.get("eyeBlinkRight", 0)) if blend else (1, 1)
    reliable = abs(pose.yaw) < 50 and min(l_open, r_open) > 0.35
    return Gaze(
        yaw=pose.yaw + eye_yaw, pitch=pose.pitch + eye_pitch, eye_yaw=eye_yaw, eye_pitch=eye_pitch,
        origin_r=P[R_EYE["iris"], :2].copy(), origin_l=P[L_EYE["iris"], :2].copy(), reliable=reliable,
    )


def geometry_metrics(P: np.ndarray) -> dict[str, float]:
    def d(a, b):
        return float(np.linalg.norm(P[a, :2] - P[b, :2]))
    r_c = (P[R_EYE["outer"], :2] + P[R_EYE["inner"], :2]) / 2
    l_c = (P[L_EYE["outer"], :2] + P[L_EYE["inner"], :2]) / 2
    iod = float(np.linalg.norm(r_c - l_c)) or 1.0
    m = {
        "face_width": d(CHEEK_R, CHEEK_L) / iod,
        "face_height": d(FOREHEAD, CHIN) / iod,
        "eye_width_r": d(R_EYE["outer"], R_EYE["inner"]) / iod,
        "eye_width_l": d(L_EYE["outer"], L_EYE["inner"]) / iod,
        "nose_width": d(NOSE_R, NOSE_L) / iod,
        "nose_length": d(168, NOSE_TIP) / iod,
        "mouth_width": d(MOUTH_R, MOUTH_L) / iod,
        "mouth_open": d(LIP_TOP, LIP_BOTTOM) / max(d(MOUTH_R, MOUTH_L), 1e-6),
        "brow_eye_r": d(BROW_R, R_EYE["top"]) / iod,
        "brow_eye_l": d(BROW_L, L_EYE["top"]) / iod,
        "chin_mouth": d(LIP_BOTTOM, CHIN) / iod,
    }
    m["aspect"] = m["face_height"] / max(m["face_width"], 1e-6)
    m["iod_px"] = iod
    return m


def canonical_mesh(P: np.ndarray) -> np.ndarray:
    X = P[:, :3].astype(np.float64)
    r_c = (X[R_EYE["outer"], :2] + X[R_EYE["inner"], :2]) / 2
    l_c = (X[L_EYE["outer"], :2] + X[L_EYE["inner"], :2]) / 2
    ax = l_c - r_c
    iod = float(np.linalg.norm(ax)) or 1.0
    ang = np.arctan2(ax[1], ax[0])
    c, s = np.cos(-ang), np.sin(-ang)
    xy = (X[:, :2] - (r_c + l_c) / 2) @ np.array([[c, -s], [s, c]]).T
    z = X[:, 2] - X[:, 2].mean()
    return np.column_stack([xy / iod, z / iod]).astype(np.float32)


def normalized_shape(P: np.ndarray) -> np.ndarray:
    X = P[:, :2].astype(np.float64)
    X = X - X.mean(0)
    ax = X[L_EYE["outer"]] - X[R_EYE["outer"]]
    ang = np.arctan2(ax[1], ax[0])
    c, s = np.cos(-ang), np.sin(-ang)
    X = X @ np.array([[c, -s], [s, c]]).T
    return X / (np.sqrt((X ** 2).sum(1).mean()) or 1.0)


def procrustes_distance(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = normalized_shape(A), normalized_shape(B)
    U, _, Vt = np.linalg.svd(b.T @ a)
    Rm = U @ Vt
    b2 = b @ Rm
    s = (a * b2).sum() / (b2 * b2).sum()
    b2 *= s
    return float(np.sqrt(((a - b2) ** 2).sum(1).mean())), b2
