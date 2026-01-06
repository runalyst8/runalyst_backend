import json
import argparse
import numpy as np
from scipy.signal import savgol_filter

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
    J = np.squeeze(J)  # handles (1,J,3)
    if J.ndim == 1:
        if J.size % 3 != 0:
            raise ValueError(f"joints3d is 1D but not divisible by 3: shape={J.shape}")
        J = J.reshape(-1, 3)
    if J.ndim != 2 or J.shape[1] != 3:
        raise ValueError(f"Unexpected joints3d shape: {J.shape}")
    return J

def pick_ankle_indices(nj, li=None, ri=None):
    if li is not None and ri is not None:
        return li, ri
    if nj == 24:      # common SMPL-24
        return 7, 8
    if nj == 17:      # COCO-17
        return 15, 16
    raise ValueError(f"Unknown joint count {nj}. Pass --li and --ri explicitly.")

def extract_y(rows, vertical_axis=1, li=None, ri=None):
    t, yL, yR = [], [], []
    li_i = ri_i = None
    for r in rows:
        nlf = r["nlf"]
        if "joints3d" not in nlf:
            continue
        J = to_joints_array(nlf["joints3d"])
        if li_i is None:
            li_i, ri_i = pick_ankle_indices(J.shape[0], li, ri)
        t.append(float(r["timestamp_sec"]))
        yL.append(float(J[li_i, vertical_axis]))
        yR.append(float(J[ri_i, vertical_axis]))
    if len(t) < 10:
        raise RuntimeError("Not enough frames with joints3d.")
    return np.array(t), np.array(yL), np.array(yR), (li_i, ri_i)

def smooth(y, win=21, poly=2):
    if len(y) < win:
        return y
    if win % 2 == 0:
        win += 1
    # ensure win < len(y) and odd
    if win >= len(y):
        win = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
    return savgol_filter(y, win, poly)

def derivative(y, t):
    dy = np.diff(y)
    dt = np.diff(t)
    return np.r_[dy[0] / max(dt[0], 1e-6), dy / np.clip(dt, 1e-6, None)]

def rising_zero_crossings(dy, t):
    # derivative negative -> positive => dy[i-1] < 0 and dy[i] >= 0
    idx = np.where((dy[1:] >= 0) & (dy[:-1] < 0))[0] + 1
    return t[idx], idx

def apply_refractory(event_times, min_interval_s=0.25):
    kept = []
    last = -1e18
    for tt in event_times:
        if tt - last >= min_interval_s:
            kept.append(tt)
            last = tt
    return np.array(kept, dtype=np.float64)

def cadence_spm(event_times):
    # steps per minute from first->last
    if len(event_times) < 2:
        return None
    duration_min = (event_times[-1] - event_times[0]) / 60.0
    if duration_min <= 0:
        return None
    # number of intervals between events
    return (len(event_times) - 1) / duration_min

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--vertical-axis", type=int, default=1, help="0=x,1=y,2=z")
    ap.add_argument("--li", type=int, default=None)
    ap.add_argument("--ri", type=int, default=None)

    ap.add_argument("--win", type=int, default=21, help="SavGol window (odd)")
    ap.add_argument("--poly", type=int, default=2)
    ap.add_argument("--min-step-interval", type=float, default=0.25,
                    help="seconds; prevents double-counting noise")

    # Optional: only count minima near 'ground'
    ap.add_argument("--use-ground-filter", action="store_true")
    ap.add_argument("--ground-quantile", type=float, default=0.10,
                    help="ground estimate quantile of min(yL,yR)")
    ap.add_argument("--ground-margin", type=float, default=40.0,
                    help="only count events where y <= ground + margin (your units)")
    ap.add_argument("--y-down", action="store_true",
                    help="If y increases downward (image coords), flip sign so minima correspond to lowest foot.")
    args = ap.parse_args()

    rows = load_ok_rows(args.jsonl)
    t, yL, yR, (li, ri) = extract_y(rows, args.vertical_axis, args.li, args.ri)

    # If y is "downwards" (pixels), flip to make "down" negative so minima still mean "lowest"
    if args.y_down:
        yL = -yL
        yR = -yR

    yL_s = smooth(yL, args.win, args.poly)
    yR_s = smooth(yR, args.win, args.poly)

    dyL = derivative(yL_s, t)
    dyR = derivative(yR_s, t)

    evL_t, evL_idx = rising_zero_crossings(dyL, t)
    evR_t, evR_idx = rising_zero_crossings(dyR, t)

    # Optional ground filter: keep only minima close to ground
    if args.use_ground_filter:
        low = np.minimum(yL_s, yR_s)
        ground = np.quantile(low, args.ground_quantile)
        keepL = yL_s[evL_idx] <= (ground + args.ground_margin)
        keepR = yR_s[evR_idx] <= (ground + args.ground_margin)
        evL_t = evL_t[keepL]
        evR_t = evR_t[keepR]
    else:
        ground = None

    evL_t = apply_refractory(evL_t, args.min_step_interval)
    evR_t = apply_refractory(evR_t, args.min_step_interval)

    cadL = cadence_spm(evL_t)  # left strikes/min (per-leg)
    cadR = cadence_spm(evR_t)

    # total cadence (steps/min) using both legs
    all_events = np.sort(np.r_[evL_t, evR_t])
    cadTotal = cadence_spm(all_events)

    print("ankle indices (L,R):", (li, ri))
    if ground is not None:
        print("estimated ground (in flipped space if y-down):", float(ground))
    print("events: L =", len(evL_t), "R =", len(evR_t), "Total =", len(all_events))
    print("Left cadence  (events/min):", cadL)
    print("Right cadence (events/min):", cadR)
    print("Total cadence (steps/min):", cadTotal)

if __name__ == "__main__":
    main()
