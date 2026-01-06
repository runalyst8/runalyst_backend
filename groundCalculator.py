import json, numpy as np
import matplotlib.pyplot as plt

def load_ok_rows(path):
    rows=[]
    for line in open(path,"r",encoding="utf-8"):
        r=json.loads(line)
        if r.get("ok") and r.get("timestamp_sec") is not None:
            rows.append(r)
    rows.sort(key=lambda x: x["timestamp_sec"])
    return rows

import numpy as np

def get_feet(nlf):
    if "joints3d" not in nlf:
        raise KeyError("nlf has no joints3d")

    J = np.asarray(nlf["joints3d"], dtype=np.float32)
    J = np.squeeze(J)  # removes (1,*,*) -> (*,*)

    # If flattened like (72,), try reshape to (-1, 3)
    if J.ndim == 1:
        if J.size % 3 != 0:
            raise ValueError(f"joints3d is 1D but not divisible by 3: shape={J.shape}")
        J = J.reshape(-1, 3)

    # Now we expect (num_joints, 3)
    if J.ndim != 2 or J.shape[1] != 3:
        raise ValueError(f"Unexpected joints3d shape after squeeze: {J.shape}")

    nj = J.shape[0]

    # Choose ankle indices based on joint count (adjust if your model differs)
    if nj == 24:          # SMPL 24 joints (common): L ankle=7, R ankle=8
        li, ri = 7, 8
    elif nj == 17:        # COCO-17: L ankle=15, R ankle=16
        li, ri = 15, 16
    else:
        raise ValueError(f"Don't know ankle indices for num_joints={nj}. "
                         f"Add mapping or use joint names if provided.")

    return J[li], J[ri]

def speeds(P, t):
    dt = np.diff(t)
    dP = np.diff(P, axis=0)
    v = np.linalg.norm(dP, axis=1) / np.clip(dt, 1e-6, None)
    return np.r_[v[0], v]

def segments(contact, t):
    segs=[]
    on=False; s=None
    for i,c in enumerate(contact):
        if c and not on: on=True; s=t[i]
        elif (not c) and on: on=False; e=t[i]; segs.append((s,e,e-s))
    if on: segs.append((s,t[-1],t[-1]-s))
    return segs

def main():
    rows = load_ok_rows("mid_strike.jsonl")
    t = np.array([r["timestamp_sec"] for r in rows], dtype=np.float64)

    L=[]; R=[]
    for r in rows:
        l,rp = get_feet(r["nlf"])
        L.append(l); R.append(rp)
    L=np.vstack(L); R=np.vstack(R)
    print("L min/max per axis:", L.min(axis=0), L.max(axis=0))
    print("R min/max per axis:", R.min(axis=0), R.max(axis=0))

    # choose vertical axis (try 1 then 2)
    vertical_axis = 1

    low = np.minimum(L[:,vertical_axis], R[:,vertical_axis])
    ground = np.quantile(low, 0.05)

    hL = L[:,vertical_axis] - ground
    hR = R[:,vertical_axis] - ground
    vL = speeds(L,t)
    vR = speeds(R,t)

    # thresholds (tune)
    h_thr = 0.03
    v_thr = 0.3

    cL = (hL < h_thr) & (vL < v_thr)
    cR = (hR < h_thr) & (vR < v_thr)

    segL = segments(cL,t)
    segR = segments(cR,t)

    print("ground:", ground)
    print("Left stance segments (start,end,dur):", segL[:10])
    print("Right stance segments (start,end,dur):", segR[:10])
    if segL: print("Left mean GCT:", np.mean([d for *_,d in segL]))
    if segR: print("Right mean GCT:", np.mean([d for *_,d in segR]))

    # quick debug plot
    plt.figure()
    plt.plot(t, hL, label="hL")
    plt.plot(t, hR, label="hR")
    plt.axhline(h_thr, linestyle="--")
    plt.legend()
    plt.title("Foot height above ground")
    plt.show()

if __name__ == "__main__":
    main()
