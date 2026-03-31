"""
Usage:
  python half_cycle_stats.py your_file.jsonl
  python half_cycle_stats.py your_file.jsonl --label "Runner Name"
  python half_cycle_stats.py your_file.jsonl --smooth-window 11 --prominence 15

NOTE: FPS is hardcoded to 60. give the fps parameter when calling from outside.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter, find_peaks


FPS = 60.0   # ← change this when you have the real FPS from cv2/ffprobe
PELVIS_IDX = 0
L_ANKLE_IDX = 7
R_ANKLE_IDX = 8


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


def get_joint(frames, joint_idx):
    """Return (N, 3) array of 3D coords for a joint across all frames."""
    return np.array([f["joints_3d"][0][0][joint_idx] for f in frames])


def label_peaks_lr(peaks, l_foot_y_sm, r_foot_y_sm):
    """
    At each peak (pelvis lowest = max stance load), the foot with higher Y
    (lower in space) is on the ground. Label that peak with that foot.
    The following trough will be attributed to that foot's push-off.
    """
    labels = []
    for pidx in peaks:
        foot = 'L' if l_foot_y_sm[pidx] > r_foot_y_sm[pidx] else 'R'
        labels.append(foot)
    return labels


def plot_signals(ts, pelvis_raw, pelvis_sm, l_foot_sm, r_foot_sm,
                 peaks, troughs, foot_labels, label):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#04080f')
    for ax in axes:
        ax.set_facecolor('#080f1c')
        ax.tick_params(labelcolor='#64748b', labelsize=8)
        ax.spines[['top', 'right', 'bottom', 'left']].set_color('#1e3554')
        ax.grid(color='#1e3554', linewidth=0.4, linestyle=':')

    # ── top: pelvis ───────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ts, pelvis_raw, color='#475569', lw=0.8, alpha=0.4, label='Pelvis raw')
    ax.plot(ts, pelvis_sm,  color='#f97316', lw=1.8,            label='Pelvis smoothed')

    l_peaks = [p for p, f in zip(peaks, foot_labels) if f == 'L']
    r_peaks = [p for p, f in zip(peaks, foot_labels) if f == 'R']
    if l_peaks:
        ax.scatter(ts[l_peaks], pelvis_sm[l_peaks], color='#4ade80', s=70, zorder=5,
                   marker='o', edgecolors='#000', lw=0.8, label='Peak — L ankle grounded')
    if r_peaks:
        ax.scatter(ts[r_peaks], pelvis_sm[r_peaks], color='#f472b6', s=70, zorder=5,
                   marker='o', edgecolors='#000', lw=0.8, label='Peak — R ankle grounded')
    ax.scatter(ts[troughs], pelvis_sm[troughs], color='#38bdf8', s=55, zorder=5,
               marker='v', edgecolors='#000', lw=0.8, label='Troughs (push-off / flight)')

    ax.set_ylabel('Pelvis Y (down = +)', color='#94a3b8', fontsize=9)
    ax.legend(fontsize=8, loc='upper left', facecolor='#0c1526',
              edgecolor='#1e3554', labelcolor='#94a3b8')

    # ── bottom: foot Y signals ────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(ts, l_foot_sm, color='#4ade80', lw=1.8, label='L ankle Y smoothed')
    ax.plot(ts, r_foot_sm, color='#f472b6', lw=1.8, label='R ankle Y smoothed')
    ax.scatter(ts[peaks], l_foot_sm[peaks], color='#4ade80', s=40, marker='|',
               zorder=5, label='at peak (L)')
    ax.scatter(ts[peaks], r_foot_sm[peaks], color='#f472b6', s=40, marker='|',
               zorder=5, label='at peak (R)')

    ax.set_ylabel('Foot Y (down = +)', color='#94a3b8', fontsize=9)
    ax.set_xlabel('Time (s)', color='#64748b', fontsize=9)
    ax.legend(fontsize=8, loc='upper left', facecolor='#0c1526',
              edgecolor='#1e3554', labelcolor='#94a3b8')

    fig.suptitle(f'Pelvis + Foot Y — {label}  (green=L  pink=R)',
                 color='#e2e8f0', fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.show()


def cadence_and_step_vertical_comparison(path: str, label: str, smooth_window: int, prominence: float, min_distance: int, fps: float = FPS):
    frames   = load_jsonl(path)
    n        = len(frames)
    ts       = np.array([f["frame_index"] / fps for f in frames])

    pelvis      = get_joint(frames, PELVIS_IDX)        # (N, 3)
    l_foot_y_sm = smooth(get_joint(frames, L_ANKLE_IDX)[:, 1], smooth_window)
    r_foot_y_sm = smooth(get_joint(frames, R_ANKLE_IDX)[:, 1], smooth_window)
    pelvis_y    = pelvis[:, 1]
    p_sm        = smooth(pelvis_y, smooth_window)

    peaks,   _ = find_peaks( p_sm, distance=min_distance, prominence=prominence)
    troughs, _ = find_peaks(-p_sm, distance=min_distance, prominence=prominence)

    # Label each peak by which foot is lower at that frame
    foot_labels = label_peaks_lr(peaks, l_foot_y_sm, r_foot_y_sm)

    # For each peak, find the FOLLOWING trough (its push-off)
    def following_trough(pidx):
        after = troughs[troughs > pidx]
        return after[0] if len(after) else None

    # 3D excursion: pelvis distance from peak to its following trough
    def excursion_3d(pidx):
        tidx = following_trough(pidx)
        if tidx is None:
            return None
        return float(abs(pelvis[pidx][1] - pelvis[tidx][1]))

    sep = "─" * 95
    print(f"\n{'═' * 95}")
    print(f"  PELVIS HALF-CYCLE STATS  —  {label}")
    print(f"{'═' * 95}")
    print(f"  File         : {path}")
    print(f"  Frames       : {n}  |  Duration: {ts[-1]:.3f} s  |  FPS: {fps} (hardcoded)")
    print(f"  Smooth window: {smooth_window}  |  Prominence: {prominence}  |  Min distance: {min_distance} frames")
    print(f"  Peaks detected   : {len(peaks)}  (frames: {peaks.tolist()})")
    print(f"  Troughs detected : {len(troughs)}  (frames: {troughs.tolist()})")
    print(f"  Logic: foot labeled at PEAK (pelvis lowest) → following trough = that foot's push-off")

    if len(peaks) < 2:
        print("\n  ⚠  Fewer than 2 peaks detected — cannot compute half-cycles.\n")
        return

    step_durations = np.diff(ts[peaks])
    total_steps = len(peaks) - 1
    time_span   = ts[peaks[-1]] - ts[peaks[0]]
    cadence     = float(total_steps / time_span * 60)

    print(f"  Cadence          : {cadence:.1f} steps/min  ({total_steps} steps over {time_span:.3f}s)\n")
    print(f"  {sep}")

    col_w   = [4, 6, 10, 8, 14, 14, 16, 16]
    headers = ["#", "Foot", "Peak frame", "t (s)", "Next peak (s)", "Half-cycle (s)",
               "Foll. trough", "Y Excursion"]
    print("  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
    print(f"  {sep}")

    excursions_L, excursions_R = [], []
    step_durations_L, step_durations_R = [], []

    for i, pidx in enumerate(peaks):
        foot   = foot_labels[i]
        next_t = f"{ts[peaks[i+1]]:.3f}" if i < len(peaks) - 1 else "—"
        hc     = f"{step_durations[i]:.3f}" if i < len(step_durations) else "—"
        tidx   = following_trough(pidx)
        trough_str = str(tidx) if tidx is not None else "—"
        exc    = excursion_3d(pidx)
        exc_str = f"{exc:.2f}" if exc is not None else "—"

        if exc is not None:
            if foot == 'L':
                excursions_L.append(exc)
                if i < len(step_durations): step_durations_L.append(step_durations[i])
            else:
                excursions_R.append(exc)
                if i < len(step_durations): step_durations_R.append(step_durations[i])

        row = [str(i+1), foot, str(pidx), f"{ts[pidx]:.3f}", next_t, hc, trough_str, exc_str]
        print("  " + "  ".join(v.ljust(col_w[j]) for j, v in enumerate(row)))

    print(f"  {sep}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_exc = excursions_L + excursions_R
    print(f"\n  ── Overall ──────────────────────────────────────────────────────────────")
    print(f"  Steps       : {len(step_durations)}")
    print(f"  Mean step time    : {step_durations.mean()*1000:.0f} ms  (std: {step_durations.std()*1000:.1f} ms)")
    print(f"  Cadence           : {cadence:.1f} steps/min")
    if all_exc:
        print(f"  Mean Y excursion : {np.mean(all_exc):.2f}  (std: {np.std(all_exc):.2f})")

    print(f"\n  ── L foot vs R foot push-off ────────────────────────────────────────────")
    print(f"  {'Metric':<30}  {'L foot':>12}  {'R foot':>12}  {'diff':>10}")
    print(f"  {'─'*68}")

    exc_L = np.mean(excursions_L) if excursions_L else None
    exc_R = np.mean(excursions_R) if excursions_R else None
    hc_L  = np.mean(step_durations_L) if step_durations_L else None
    hc_R  = np.mean(step_durations_R) if step_durations_R else None

    exc_L_str = f"{exc_L:.2f}" if exc_L else "—"
    exc_R_str = f"{exc_R:.2f}" if exc_R else "—"
    exc_diff  = f"{exc_L - exc_R:+.2f}" if (exc_L and exc_R) else "—"
    hc_L_str  = f"{hc_L*1000:.0f} ms"  if hc_L else "—"
    hc_R_str  = f"{hc_R*1000:.0f} ms"  if hc_R else "—"
    hc_diff   = f"{(hc_L - hc_R)*1000:+.0f} ms" if (hc_L and hc_R) else "—"

    print(f"  {'Mean Y excursion':<30}  {exc_L_str:>12}  {exc_R_str:>12}  {exc_diff:>10}")
    print(f"  {'Mean step duration':<30}  {hc_L_str:>12}  {hc_R_str:>12}  {hc_diff:>10}")
    print(f"  {'Steps detected':<30}  {len(excursions_L):>12}  {len(excursions_R):>12}")

    print()


    # stride length
    # At each peak, one ankle is on the ground (determined by foot_labels).
    # We collect the 3D position of that grounded ankle at every peak it touches down.
    # Stride length = 3D Euclidean distance between consecutive same-foot touchdowns.
    l_ankle = get_joint(frames, L_ANKLE_IDX)  # (N, 3)
    r_ankle = get_joint(frames, R_ANKLE_IDX)  # (N, 3)

    # one (3,) array per touchdown of that foot
    l_contact_coords = np.array([l_ankle[p] for p, f in zip(peaks, foot_labels) if f == 'L'])
    r_contact_coords = np.array([r_ankle[p] for p, f in zip(peaks, foot_labels) if f == 'R'])

    # distance between consecutive same-foot contacts
    l_strides = [float(np.linalg.norm(l_contact_coords[i+1] - l_contact_coords[i]))
                 for i in range(len(l_contact_coords) - 1)] if len(l_contact_coords) >= 2 else []
                 
    r_strides = [float(np.linalg.norm(r_contact_coords[i+1] - r_contact_coords[i]))
                 for i in range(len(r_contact_coords) - 1)] if len(r_contact_coords) >= 2 else []

    mean_stride_L = float(np.mean(l_strides)) if l_strides else None
    mean_stride_R = float(np.mean(r_strides)) if r_strides else None

    print(f"\n  ── Stride length (grounded ankle, same-foot peak → next same-foot peak) ──")
    print(f"  {'Metric':<30}  {'L foot':>12}  {'R foot':>12}  {'diff':>10}")
    print(f"  {'─'*68}")
    sl_L_str  = f"{mean_stride_L:.2f}" if mean_stride_L else "—"
    sl_R_str  = f"{mean_stride_R:.2f}" if mean_stride_R else "—"
    sl_diff   = f"{mean_stride_L - mean_stride_R:+.2f}" if (mean_stride_L and mean_stride_R) else "—"
    print(f"  {'Mean stride length':<30}  {sl_L_str:>12}  {sl_R_str:>12}  {sl_diff:>10}")
    print(f"  {'Individual strides (L)':<30}  {[round(x, 1) for x in l_strides]}")
    print(f"  {'Individual strides (R)':<30}  {[round(x, 1) for x in r_strides]}")

    plot_signals(ts, pelvis_y, p_sm, l_foot_y_sm, r_foot_y_sm,
                 peaks, troughs, foot_labels, label)
    
    return cadence, exc_L, exc_R, hc_L, hc_R, mean_stride_L, mean_stride_R, peaks, foot_labels



def main():
    parser = argparse.ArgumentParser(
        description="Pelvis half-cycle stats with L/R foot push-off excursion comparison"
    )
    parser.add_argument("jsonl_file")
    parser.add_argument("--label",         default="Runner")
    parser.add_argument("--smooth-window", type=int,   default=11)
    parser.add_argument("--prominence",    type=float, default=15.0)
    parser.add_argument("--min-distance",  type=int,   default=10)
    parser.add_argument("--fps",           type=float, default=FPS, help="Frames per second (default: 60)")
    args = parser.parse_args()

    cadence_and_step_vertical_comparison(
        path          = args.jsonl_file,
        label         = args.label,
        smooth_window = args.smooth_window,
        prominence    = args.prominence,
        min_distance  = args.min_distance,
    )


if __name__ == "__main__":
    main()


# when using the function in other files, you can call analyze() directly with the appropriate parameters, and it will return the cadence, mean excursions for left and right foot, mean step durations for left and right foot, peak frames, and foot labels.
# also comment the plotting command
# from pelvis_analysis import cadence_and_step_vertical_comparison
# cadence, exc_L, exc_R, hc_L, hc_R, mean_stride_L, mean_stride_R, peaks, foot_labels = cadence_and_step_vertical_comparison(
#     path="your_file.jsonl",
#     label="Runner Name",
#     smooth_window=11,
#     prominence=15.0,
#     min_distance=10
# )
