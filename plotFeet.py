import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # ✅ headless backend for SSH
import matplotlib.pyplot as plt


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


def pick_ankle_indices(nj, li_arg=None, ri_arg=None):
    if li_arg is not None and ri_arg is not None:
        return li_arg, ri_arg
    if nj == 24:      # SMPL-24 (common)
        return 7, 8
    if nj == 17:      # COCO-17
        return 15, 16
    raise ValueError(f"Unknown joint count {nj}. Provide --li and --ri explicitly.")


def extract_feet(rows, li_arg=None, ri_arg=None):
    t, L, R = [], [], []
    li = ri = None

    for r in rows:
        nlf = r["nlf"]

        if "joints3d" in nlf:
            J = to_joints_array(nlf["joints3d"])
            if li is None:
                li, ri = pick_ankle_indices(J.shape[0], li_arg, ri_arg)
            L.append(J[li])
            R.append(J[ri])

        elif "joints" in nlf:
            L.append(np.asarray(nlf["joints"]["left_ankle"], dtype=np.float32))
            R.append(np.asarray(nlf["joints"]["right_ankle"], dtype=np.float32))
        else:
            continue

        t.append(float(r["timestamp_sec"]))

    if not t:
        raise RuntimeError("No usable frames found (need ok=True and joints3d or named joints).")

    return np.array(t), np.vstack(L), np.vstack(R), (li, ri)


def save_xyz_plots(t, P, title_prefix, outdir, dpi=150):
    axis_names = ["X", "Y", "Z"]
    for ax_i, ax_name in enumerate(axis_names):
        fig = plt.figure()
        plt.plot(t, P[:, ax_i])
        plt.xlabel("time (s)")
        plt.ylabel(f"{ax_name} (your units)")
        plt.title(f"{title_prefix} {ax_name} vs time")
        plt.tight_layout()

        fname = outdir / f"{title_prefix.lower().replace(' ', '_')}_{ax_name.lower()}_vs_time.png"
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)


def save_xz_topdown(L, R, outdir, dpi=150):
    fig = plt.figure()
    plt.plot(L[:, 0], L[:, 2], label="Left")
    plt.plot(R[:, 0], R[:, 2], label="Right")
    plt.xlabel("X (your units)")
    plt.ylabel("Z (your units)")
    plt.title("Foot trajectory (X–Z)")
    plt.legend()
    plt.tight_layout()

    fname = outdir / "feet_trajectory_xz.png"
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to out_nlf.jsonl")
    ap.add_argument("--outdir", default="plots_out", help="Output directory for PNGs")
    ap.add_argument("--dpi", type=int, default=150, help="DPI for saved figures")
    ap.add_argument("--li", type=int, default=None, help="Left ankle joint index (if joints3d)")
    ap.add_argument("--ri", type=int, default=None, help="Right ankle joint index (if joints3d)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_ok_rows(args.jsonl)
    t, L, R, (li, ri) = extract_feet(rows, args.li, args.ri)

    print("Loaded frames:", len(t))
    print("Left/Right joint indices:", (li, ri))
    print("L min/max:", L.min(axis=0), L.max(axis=0))
    print("R min/max:", R.min(axis=0), R.max(axis=0))
    print("Saving plots to:", outdir.resolve())

    save_xyz_plots(t, L, "Left foot", outdir, dpi=args.dpi)
    save_xyz_plots(t, R, "Right foot", outdir, dpi=args.dpi)
    save_xz_topdown(L, R, outdir, dpi=args.dpi)

    print("Done. Files:")
    for p in sorted(outdir.glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
