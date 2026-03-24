"""
Forward trunk lean analysis for SMPL 24-joint 3D pose data.
Computes three segment angles to distinguish where lean originates:

    Angle 1 — global   : Pelvis[0]  → Neck[12]    overall trunk inclination
    Angle 2 — lower    : Pelvis[0]  → Spine2[6]   pelvic + lumbar contribution
    Angle 3 — upper    : Spine2[6]  → Neck[12]    thoracic contribution

Interpreting the three together:
    angle2 large, angle3 small  →  lean driven from the hip/pelvis (good running posture)
    angle3 large, angle2 small  →  lean driven from thoracic rounding (upper-back collapse)
    both large                  →  lean distributed across the whole spine

Usage:
    python trunk_lean_analysis.py your_file.jsonl
    python trunk_lean_analysis.py your_file.jsonl --label "Ceren" --frame-step 2 --fps 64

Coordinate system (verified from data):
    X  ->  running direction (sign auto-detected from pelvis trajectory)
    Y  ->  vertical (+ = down)
    Z  ->  lateral

Gravity assumed = (0, 1, 0).
All angles computed in the sagittal plane (X-Y), Z zeroed out.
Positive = leaning forward. Negative = leaning backward.
"""
#önemli not:  sonuç 3 ten küçük ise fazla dik, 3- 10 arası normal, 14+ belirgin kambur oluyor.

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

FPS        = 64.0
PELVIS_IDX = 0
SPINE2_IDX = 6
NECK_IDX   = 12
UPWARD     = np.array([0.0, -1.0, 0.0])  # opposite of gravity (+Y down)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    frames.sort(key=lambda x: x["frame_index"])
    return frames


def smooth(signal: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    w = min(window, len(signal))
    w = w if w % 2 else w - 1
    w = max(w, 5)
    return savgol_filter(signal, w, poly)


def _segment_angle(a: np.ndarray, b: np.ndarray, run_dir: int) -> float:
    """
    Forward lean angle of the segment a->b in the sagittal plane (X-Y).
    Positive = tip of segment (b) displaced in running direction vs base (a).
    """
    vec = b - a
    vec_sag = np.array([vec[0], vec[1], 0.0])
    norm = np.linalg.norm(vec_sag)
    if norm < 1e-6:
        return 0.0
    vec_sag /= norm

    cos_a = np.clip(np.dot(vec_sag, UPWARD), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_a))

    # positive when the tip is forward of the base
    if vec[0] * run_dir < 0:
        angle = -angle
    return angle


# ── core computation ──────────────────────────────────────────────────────────

def compute_forward_trunk_lean(
    frames:     list[dict],
    frame_step: int = 1,
    pelvis_idx: int = PELVIS_IDX,
    spine2_idx: int = SPINE2_IDX,
    neck_idx:   int = NECK_IDX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward trunk lean angles for each selected frame.

    Parameters
    ----------
    frames      : list of frame dicts loaded from JSONL
    frame_step  : 1 = every frame, 2 = every other frame, etc.
    pelvis_idx  : SMPL joint index for pelvis (default 0)
    spine2_idx  : SMPL joint index for Spine2 (default 6)
    neck_idx    : SMPL joint index for neck   (default 12)

    Returns
    -------
    frame_indices  : (N,) int array
    global_angles  : (N,) Pelvis -> Neck        overall trunk lean
    lower_angles   : (N,) Pelvis -> Spine2      pelvic + lumbar contribution
    upper_angles   : (N,) Spine2 -> Neck        thoracic contribution

    All angles in degrees. Positive = forward lean.
    """
    # Auto-detect running direction from pelvis X trajectory
    first_x = frames[0]["joints_3d"][0][0][pelvis_idx][0]
    last_x  = frames[-1]["joints_3d"][0][0][pelvis_idx][0]
    run_dir = +1 if last_x > first_x else -1
    print(f"  Running direction : {'+ X' if run_dir == 1 else '- X'}  "
          f"(pelvis X: {first_x:.1f} -> {last_x:.1f})")
    print(f"  Forward lean sign : neck.x {'>' if run_dir == 1 else '<'} "
          f"pelvis.x = positive angle")

    selected      = frames[::frame_step]
    frame_indices = np.array([f["frame_index"] for f in selected])
    global_angles = np.zeros(len(selected))
    lower_angles  = np.zeros(len(selected))
    upper_angles  = np.zeros(len(selected))

    for i, f in enumerate(selected):
        joints = np.array(f["joints_3d"][0][0])   # (24, 3)
        pelvis = joints[pelvis_idx]
        spine2 = joints[spine2_idx]
        neck   = joints[neck_idx]

        global_angles[i] = _segment_angle(pelvis, neck,   run_dir)
        lower_angles[i]  = _segment_angle(pelvis, spine2, run_dir)
        upper_angles[i]  = _segment_angle(spine2, neck,   run_dir)

    return frame_indices, global_angles, lower_angles, upper_angles


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_trunk_lean(
    frame_indices: np.ndarray,
    global_angles: np.ndarray,
    lower_angles:  np.ndarray,
    upper_angles:  np.ndarray,
    fps:           float,
    label:         str,
    smooth_window: int = 11,
) -> None:
    ts = frame_indices / fps

    g_sm = smooth(global_angles, smooth_window)
    l_sm = smooth(lower_angles,  smooth_window)
    u_sm = smooth(upper_angles,  smooth_window)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.patch.set_facecolor('#04080f')
    for ax in axes:
        ax.set_facecolor('#080f1c')
        ax.tick_params(labelcolor='#64748b', labelsize=8)
        ax.spines[['top', 'right', 'bottom', 'left']].set_color('#1e3554')
        ax.grid(color='#1e3554', linewidth=0.4, linestyle=':')

    C_GLOBAL = '#f97316'   # orange
    C_LOWER  = '#f472b6'   # pink
    C_UPPER  = '#4ade80'   # green

    for ax, xs, xlabel in zip(axes, [frame_indices, ts], ['Frame index', 'Time (s)']):
        # raw (faint)
        ax.plot(xs, global_angles, color=C_GLOBAL, lw=0.6, alpha=0.2)
        ax.plot(xs, lower_angles,  color=C_LOWER,  lw=0.6, alpha=0.2)
        ax.plot(xs, upper_angles,  color=C_UPPER,  lw=0.6, alpha=0.2)

        # smoothed + labelled
        ax.plot(xs, g_sm, color=C_GLOBAL, lw=1.8,
                label=f'Global  Pelvis->Neck    mean={np.mean(global_angles):+.1f}')
        ax.plot(xs, l_sm, color=C_LOWER,  lw=1.8,
                label=f'Lower   Pelvis->Spine2  mean={np.mean(lower_angles):+.1f}')
        ax.plot(xs, u_sm, color=C_UPPER,  lw=1.8,
                label=f'Upper   Spine2->Neck    mean={np.mean(upper_angles):+.1f}')

        ax.axhline(0, color='#334155', lw=0.8)
        ax.axhspan(5, 8, alpha=0.05, color='#38bdf8', label='Elite global range 5-8')

        ax.set_ylabel('Forward lean (deg)', color='#94a3b8', fontsize=9)
        ax.set_xlabel(xlabel, color='#64748b', fontsize=9)
        ax.legend(fontsize=8, loc='upper right', facecolor='#0c1526',
                  edgecolor='#1e3554', labelcolor='#94a3b8')

    fig.suptitle(
        f'Forward trunk lean  --  {label}\n'
        f'orange = global   pink = lower (pelvis/lumbar)   green = upper (thoracic)',
        color='#e2e8f0', fontsize=11, fontweight='bold',
    )
    fig.tight_layout()
    plt.show()


# ── main analysis function ────────────────────────────────────────────────────

def analyze_forward_trunk_lean(
    path:          str,
    label:         str   = 'Runner',
    frame_step:    int   = 1,
    fps:           float = FPS,
    smooth_window: int   = 11,
    plot:          bool  = True,
) -> dict:
    """
    Full pipeline: load -> compute -> print summary -> (optionally) plot.

    Returns
    -------
    dict with keys:
        frame_indices,
        global_angles, lower_angles, upper_angles,   -- per-frame arrays
        mean_global, std_global, min_global, max_global,
        mean_lower,  std_lower,  min_lower,  max_lower,
        mean_upper,  std_upper,  min_upper,  max_upper,
    """
    frames = load_jsonl(path)

    frame_indices, global_angles, lower_angles, upper_angles = \
        compute_forward_trunk_lean(frames, frame_step=frame_step)

    def stats(arr):
        return (float(np.mean(arr)), float(np.std(arr)),
                float(np.min(arr)),  float(np.max(arr)))

    mg, sg, ning, xg = stats(global_angles)
    ml, sl, nil, xl  = stats(lower_angles)
    mu, su, niu, xu  = stats(upper_angles)

    sep = '-' * 68
    print(f"\n{'=' * 68}")
    print(f"  FORWARD TRUNK LEAN  --  {label}")
    print(f"{'=' * 68}")
    print(f"  File        : {path}")
    print(f"  Total frames: {len(frames)}  |  Used: {len(frame_indices)}  "
          f"(step={frame_step})")
    print(f"  FPS         : {fps}  |  Duration: {frames[-1]['frame_index']/fps:.2f} s")
    print(f"  {sep}")
    print(f"  {'Segment':<26}  {'Mean':>8}  {'Std':>7}  {'Min':>8}  {'Max':>8}")
    print(f"  {sep}")
    print(f"  {'Global  (Pelvis->Neck)':<26}  {mg:>+8.2f}  {sg:>7.2f}  "
          f"{ning:>+8.2f}  {xg:>+8.2f}")
    print(f"  {'Lower   (Pelvis->Spine2)':<26}  {ml:>+8.2f}  {sl:>7.2f}  "
          f"{nil:>+8.2f}  {xl:>+8.2f}")
    print(f"  {'Upper   (Spine2->Neck)':<26}  {mu:>+8.2f}  {su:>7.2f}  "
          f"{niu:>+8.2f}  {xu:>+8.2f}")
    print(f"  {sep}")

    # Interpret the dominant pattern
    if ml > mu + 2:
        pattern = "lean driven from hip/pelvis -- lower segment dominates (good)"
    elif mu > ml + 2:
        pattern = "lean driven from thoracic rounding -- upper segment dominates"
    else:
        pattern = "lean distributed evenly across the whole spine"
    print(f"  Pattern : {pattern}")
    print(f"  {sep}")
    print(f"  Reference: elite global lean ~5-8 deg at midstance")

    if plot:
        plot_trunk_lean(
            frame_indices, global_angles, lower_angles, upper_angles,
            fps=fps, label=label, smooth_window=smooth_window,
        )

    return dict(
        frame_indices=frame_indices,
        global_angles=global_angles,
        lower_angles=lower_angles,
        upper_angles=upper_angles,
        mean_global=mg, std_global=sg, min_global=ning, max_global=xg,
        mean_lower=ml,  std_lower=sl,  min_lower=nil,   max_lower=xl,
        mean_upper=mu,  std_upper=su,  min_upper=niu,   max_upper=xu,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Forward trunk lean analysis from SMPL JSONL pose data"
    )
    parser.add_argument("jsonl_file")
    parser.add_argument("--label",         default="Runner")
    parser.add_argument("--frame-step",    type=int,   default=1,
                        help="1=all frames, 2=every other, etc.")
    parser.add_argument("--fps",           type=float, default=FPS)
    parser.add_argument("--smooth-window", type=int,   default=11)
    args = parser.parse_args()

    analyze_forward_trunk_lean(
        path          = args.jsonl_file,
        label         = args.label,
        frame_step    = args.frame_step,
        fps           = args.fps,
        smooth_window = args.smooth_window,
    )


if __name__ == "__main__":
    main()