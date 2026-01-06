import json
import numpy as np


def load_ok_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("timestamp_sec") is not None and "nlf" in r:
                rows.append(r)
    rows.sort(key=lambda x: x["timestamp_sec"])
    return rows


def to_joints_array(joints3d):
    J = np.asarray(joints3d, dtype=np.float32)
    J = np.squeeze(J)

    if J.ndim == 1:
        J = J.reshape(-1, 3)

    if J.ndim != 2 or J.shape[1] != 3:
        raise ValueError(f"Unexpected joints3d shape {J.shape}")

    return J


def get_feet(nlf):
    J = to_joints_array(nlf["joints3d"])
    nj = J.shape[0]

    if nj == 24:      # SMPL-24
        li, ri = 7, 8
    elif nj == 17:    # COCO-17
        li, ri = 15, 16
    else:
        raise ValueError(f"Unknown joint count {nj}")

    return J[li], J[ri]


def speeds(P, t):
    dt = np.diff(t)
    dP = np.diff(P, axis=0)
    v = np.linalg.norm(dP, axis=1) / np.clip(dt, 1e-6, None)
    return np.r_[v[0], v]


def detect_contacts_root_relative(P, t, vertical_axis):
    """
    Robust contact detector for root-relative data.
    Uses:
      - local minima in vertical trajectory
      - low velocity
    """
    h = P[:, vertical_axis]
    v = speeds(P, t)

    # normalize height for stability
    h_norm = h - np.min(h)

    # thresholds (safe for running)
    h_thr = np.percentile(h_norm, 25)  # lowest 10%
    v_thr = np.percentile(v, 50)            # slowest 30%

    contact = (h_norm < h_thr) & (v < v_thr)
    return contact


def stance_segments(contact, t):
    segs = []
    on = False
    s = None

    for i, c in enumerate(contact):
        if c and not on:
            on = True
            s = t[i]
        elif not c and on:
            on = False
            e = t[i]
            segs.append((s, e, e - s))

    if on:
        segs.append((s, t[-1], t[-1] - s))

    return segs
