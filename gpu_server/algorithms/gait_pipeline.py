"""
Integrated gait analysis pipeline.

Combines three steps in one function:

    Step 1 — Step finding (pelvis oscillation)
              Finds midstance frames and labels each step L or R.
              Uses pelvis Y peaks, same logic as pelvis_analysis.py.

    Step 2 — Strike type detection (strike_analysis.py)
              Determines whether the runner is a HEEL, MID, or TOE striker
              using the normalized ankle/foot Y difference method.

    Step 3 — Initial contact detection
              Searches backward from each midstance frame to find
              the frame where the contact joint first crosses a threshold.

              Contact joint depends on strike type:
                  HEEL → ankle[7/8]
                  TOE  → foot[10/11]
                  MID  → ankle[7/8]

              Threshold = midstance joint Y − offset
                  TOE  offset: 20   (foot 20 units above ground at contact)
                  MID  offset: 40
                  HEEL offset: 80   (ankle 80 units above ground at contact)

              Coordinate system: Y increases downward, so
                  higher Y = closer to ground
                  lower  Y = higher in space

Bugs fixed vs previous version
-------------------------------
  Bug 1 — threshold was computed as (peak_joint_y - 80) - additional_offset,
           putting it 100-155 units above ground instead of 20-80.
           Fix: threshold = peak_joint_y - offset  (no intermediate base_ground)

  Bug 2 — HEEL/MID search condition was window_y <= ground_level which finds
           frames where the joint is still in the air (above threshold),
           i.e. the swing phase — exactly the opposite of initial contact.
           Fix: unified backward search (same as TOE) with jy[i] < threshold

Usage
-----
    python gait_pipeline.py file.jsonl
    python gait_pipeline.py file.jsonl --label "Ceren" --fps 64 --plot

    # import:
    from gait_pipeline import run_gait_pipeline
    result = run_gait_pipeline(path="file.jsonl", label="Ceren", fps=64.0)

    result["strike_type"]   # "TOE" / "MID" / "HEEL"
    result["contacts"]      # list of contact dicts
    result["peaks"]         # midstance frame indices
    result["foot_labels"]   # ["L", "R", "L", ...]

SMPL joints used:
    Pelvis[0], L_Ankle[7], R_Ankle[8], L_Foot[10], R_Foot[11]
"""

import json
import argparse
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import sys
import os

# pelvis_analysis.py lives in the same folder as the JSONL files.
# We insert its directory so we can import from it without copying it.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, '..', 'uploads')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# reuse helpers already defined in pelvis_analysis.py — no duplication
from pelvis_analysis import load_jsonl, smooth, label_peaks_lr
from new_strike_analysis  import compute_strike_metrics

FPS           = 64.0
PELVIS        = 0
L_ANKLE       = 7;  R_ANKLE = 8
L_FOOT        = 10; R_FOOT  = 11

SMOOTH_WINDOW = 11
PROMINENCE    = 15.0
MIN_DISTANCE  = 10
SEARCH_BACK   = 10

# How many Y units above ground level = initial contact per strike type.
# Y increases downward, so contact joint at IC is at (ground_Y - offset).
IC_OFFSETS = {
    "TOE": 15,   # toe/foot nearly on ground at contact
    "MID":  30,
    "HEEL": 80,   # ankle is higher above ground at heel contact
}


# ── private helper ────────────────────────────────────────────────
# Defined here rather than imported from pelvis_analysis so the pipeline
# is not sensitive to whether the local pelvis_analysis.get_joint takes
# (frames, idx) or (frames, idx, axis).

def _get_joint_y(frames: list[dict], joint_idx: int) -> np.ndarray:
    """Return (N,) array of Y coordinates for one joint across all frames."""
    return np.array([f["joints_3d"][0][0][joint_idx][1] for f in frames])


# ── step 1: midstance peaks ───────────────────────────────────────

def find_midstance_peaks(
    frames:        list[dict],
    smooth_window: int   = SMOOTH_WINDOW,
    prominence:    float = PROMINENCE,
    min_distance:  int   = MIN_DISTANCE,
) -> tuple[np.ndarray, list[str]]:
    """
    Find midstance frames from pelvis Y oscillation and label each L or R.

    Returns
    -------
    peaks       : array of array-indices where pelvis Y is at local max
    foot_labels : list of "L"/"R", one per peak
    """
    pelvis_y = _get_joint_y(frames, PELVIS)
    p_sm     = smooth(pelvis_y, smooth_window)
    l_ankle  = smooth(_get_joint_y(frames, L_ANKLE), smooth_window)
    r_ankle  = smooth(_get_joint_y(frames, R_ANKLE), smooth_window)

    peaks, _ = find_peaks(p_sm, distance=min_distance, prominence=prominence)
    labels   = label_peaks_lr(peaks, l_ankle, r_ankle)
    return peaks, labels


# ── step 2: strike type ───────────────────────────────────────────

def detect_strike_type(frames: list[dict]) -> dict:
    """
    Detect strike type using normalized ankle/foot Y difference method.
    Wraps compute_strike_metrics from strike_analysis.py.

    Returns dict with keys:
        overall, confidence, metrics_agree, mn_max, mn_amax,
        primary, validation
    """
    m = compute_strike_metrics(frames)
    return dict(
        overall       = m["primary"],
        confidence    = m["confidence"],
        metrics_agree = m["metrics_agree"],
        mn_max        = m["mn_max"],
        mn_amax       = m["mn_amax"],
        primary       = m["primary"],
        validation    = m["validation"],
    )


# ── step 3: initial contact detection ────────────────────────────

def find_initial_contacts(
    frames:      list[dict],
    peaks:       np.ndarray,
    foot_labels: list[str],
    strike_type: str,
    search_back: int = SEARCH_BACK,
) -> list[dict]:
    """
    Find initial contact frame for each step by searching backward
    from the midstance peak.

    Logic
    -----
    - ground_level = joint Y at midstance (this is the maximum Y = on ground)
    - threshold    = ground_level - IC_OFFSET
                     (N units above ground in real space = N less in Y)
    - search backward: find the last frame where joint Y < threshold
      (still in the air), then IC = one frame after that

    Parameters
    ----------
    frames       : list of frame dicts
    peaks        : midstance array indices from find_midstance_peaks()
    foot_labels  : "L"/"R" per peak
    strike_type  : "TOE", "MID", or "HEEL"
    search_back  : max frames to search before midstance (default 10)

    Returns
    -------
    list of dicts sorted by frame, each with:
        frame               : initial contact frame index
        side                : "L" or "R"
        midstance_frame     : midstance frame index
        frames_to_midstance : midstance_frame - frame
        contact_joint       : "foot" (TOE) or "ankle" (MID/HEEL)
        fallback            : True if threshold never crossed in window
    """
    strike_type   = strike_type.upper()
    ic_offset     = IC_OFFSETS.get(strike_type, 40)
    contact_joint = "foot" if strike_type == "TOE" else "ankle"

    # precompute smoothed Y for the relevant contact joints
    joint_y = {
        "L": smooth(_get_joint_y(frames, L_FOOT   if strike_type == "TOE" else L_ANKLE)),
        "R": smooth(_get_joint_y(frames, R_FOOT   if strike_type == "TOE" else R_ANKLE)),
    }

    contacts = []
    for peak_pos, side in zip(peaks, foot_labels):
        jy = joint_y[side]

        # ground level = joint Y at midstance (maximum Y = on the ground)
        # threshold = ground_level - offset (N units above ground in space)
        ground_level = float(jy[peak_pos])
        threshold    = ground_level - ic_offset

        # Search frames at positions -6 to -11 from midstance.
        # If jy[i] < threshold at position -6  -> ic_pos = -5  -> delta = 5  (min)
        # If jy[i] < threshold at position -11 -> ic_pos = -10 -> delta = 10 (max)
        # Fallback = peak_pos - 10 -> delta = 10
        search_start = max(0, peak_pos - 11)   # search stops here
        search_end   = max(0, peak_pos - 6)    # search starts here

        ic_pos   = max(0, peak_pos - 10)       # fallback: delta = 10
        fallback = True
        for i in range(search_end, search_start - 1, -1):
            if jy[i] < threshold:
                ic_pos   = i + 1               # delta will be between 5 and 10
                fallback = False
                break

        contacts.append(dict(
            frame               = int(ic_pos),
            side                = side,
            midstance_frame     = int(peak_pos),
            frames_to_midstance = int(peak_pos - ic_pos),
            contact_joint       = contact_joint,
            fallback            = fallback,
        ))

    contacts.sort(key=lambda x: x["frame"])
    return contacts


# ── plotting ──────────────────────────────────────────────────────

def plot_results(
    frames:      list[dict],
    peaks:       np.ndarray,
    foot_labels: list[str],
    contacts:    list[dict],
    strike_type: str,
    label:       str,
) -> None:
    n = len(frames)
    t = np.arange(n)

    l_ankle_y = smooth(_get_joint_y(frames, L_ANKLE))
    r_ankle_y = smooth(_get_joint_y(frames, R_ANKLE))
    l_foot_y  = smooth(_get_joint_y(frames, L_FOOT))
    r_foot_y  = smooth(_get_joint_y(frames, R_FOOT))

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.patch.set_facecolor('#04080f')
    for ax in axes:
        ax.set_facecolor('#080f1c')
        ax.tick_params(labelcolor='#64748b', labelsize=8)
        ax.spines[['top','right','bottom','left']].set_color('#1e3554')
        ax.grid(color='#1e3554', linewidth=0.4, linestyle=':')

    C_L, C_R = '#4ade80', '#f472b6'

    # top: ankle Y with midstance markers
    ax = axes[0]
    ax.plot(t, l_ankle_y, color=C_L, lw=1.4, label='L ankle Y')
    ax.plot(t, r_ankle_y, color=C_R, lw=1.4, label='R ankle Y')
    for peak, side in zip(peaks, foot_labels):
        yval = l_ankle_y[peak] if side == 'L' else r_ankle_y[peak]
        ax.scatter(peak, yval, color=C_L if side=='L' else C_R,
                   s=80, marker='^', zorder=5, edgecolors='#000', lw=1)
    ax.set_ylabel('Ankle Y (down=+)', color='#94a3b8', fontsize=9)
    ax.set_title(f'Midstance peaks — {label}  [{strike_type}]',
                 color='#94a3b8', fontsize=10)
    ax.legend(fontsize=8, facecolor='#0c1526',
              edgecolor='#1e3554', labelcolor='#94a3b8')

    # bottom: contact joint Y with IC markers
    ax = axes[1]
    ax.plot(t, l_foot_y,  color=C_L, lw=1.5, label='L foot Y')
    ax.plot(t, r_foot_y,  color=C_R, lw=1.5, label='R foot Y')
    ax.plot(t, l_ankle_y, color=C_L, lw=0.8, ls='--', alpha=0.4, label='L ankle Y')
    ax.plot(t, r_ankle_y, color=C_R, lw=0.8, ls='--', alpha=0.4, label='R ankle Y')

    for c in contacts:
        side = c['side']
        if c['contact_joint'] == 'foot':
            jy = l_foot_y if side == 'L' else r_foot_y
        else:
            jy = l_ankle_y if side == 'L' else r_ankle_y
        color  = C_L if side == 'L' else C_R
        marker = 'o' if not c['fallback'] else 'x'
        ax.scatter(c['frame'], jy[c['frame']],
                   color=color, s=110, marker=marker,
                   zorder=5, edgecolors='#000', lw=1.5)

    ax.set_ylabel('Joint Y (down=+)', color='#94a3b8', fontsize=9)
    ax.set_xlabel('Frame', color='#64748b', fontsize=9)
    ax.set_title('Initial contacts  (circle = reliable,  x = fallback)',
                 color='#94a3b8', fontsize=10)
    ax.legend(fontsize=8, facecolor='#0c1526',
              edgecolor='#1e3554', labelcolor='#94a3b8')

    fig.suptitle(
        f'Gait pipeline — {label}  |  strike: {strike_type}  '
        f'|  IC offset: {IC_OFFSETS.get(strike_type.upper(), "?")} Y units',
        color='#e2e8f0', fontsize=11, fontweight='bold',
    )
    fig.tight_layout()
    plt.show()


# ── printing ──────────────────────────────────────────────────────

def print_results(
    path:        str,
    label:       str,
    fps:         float,
    n_frames:    int,
    peaks:       np.ndarray,
    foot_labels: list[str],
    strike:      dict,
    contacts:    list[dict],
) -> None:
    sep  = "─" * 72
    sep2 = "═" * 72

    print(f"\n{sep2}")
    print(f"  GAIT PIPELINE  —  {label}")
    print(f"{sep2}")
    print(f"  File         : {path}")
    print(f"  Total frames : {n_frames}  |  FPS: {fps}  |  "
          f"Duration: {n_frames/fps:.2f} s")

    # step 1
    print(f"\n  {'─'*28}  STEP 1: Step finding  {'─'*18}")
    print(f"  Midstance frames detected : {len(peaks)}")
    print(f"  {'Step':>5}  {'Side':>5}  {'Midstance frame':>17}")
    print(f"  {sep}")
    for i, (p, lbl) in enumerate(zip(peaks, foot_labels)):
        print(f"  {i+1:>5}  {lbl:>5}  {p:>17}")

    # step 2
    print(f"\n  {'─'*28}  STEP 2: Strike type  {'─'*19}")
    agree_str = "agree" if strike["metrics_agree"] else "disagree"
    print(f"  Primary   (norm_diff) : {strike['mn_max']:>+.4f}  →  {strike['primary']}")
    print(f"  Validation (angle)    : {strike['mn_amax']:>+.1f}°"
          f"  →  {strike['validation']}  ({agree_str})")
    print(f"  Confidence            : {strike['confidence']}")
    print(f"  {'─'*50}")
    print(f"  Strike type           : {strike['overall']}")

    # step 3
    c_joint = contacts[0]["contact_joint"] if contacts else "—"
    n_valid = sum(1 for c in contacts if not c["fallback"])
    n_fall  = len(contacts) - n_valid
    ic_off  = IC_OFFSETS.get(strike["overall"], "?")

    print(f"\n  {'─'*26}  STEP 3: Initial contacts  {'─'*16}")
    print(f"  Contact joint : {c_joint}  "
          f"(strike={strike['overall']}, threshold offset={ic_off} Y units)")
    print(f"  {sep}")
    print(f"  {'IC frame':>9}  {'Side':>5}  {'Midstance':>10}  "
          f"{'Δ frames':>9}  {'Δ ms':>9}  {'Note':>8}")
    print(f"  {sep}")

    for c in contacts:
        note    = "fallback" if c["fallback"] else ""
        diff_ms = c["frames_to_midstance"] / fps * 1000
        print(f"  {c['frame']:>9}  {c['side']:>5}  "
              f"{c['midstance_frame']:>10}  "
              f"{c['frames_to_midstance']:>9}  "
              f"{diff_ms:>8.1f}ms  "
              f"{note:>8}")

    print(f"  {sep}")
    print(f"  Steps : {len(contacts)} total  "
          f"({n_valid} reliable, {n_fall} fallback)")

    diffs = [c["frames_to_midstance"] for c in contacts if not c["fallback"]]
    if diffs:
        print(f"  IC→midstance : mean={np.mean(diffs):.1f} fr  "
              f"std={np.std(diffs):.1f}  "
              f"range=[{min(diffs)}, {max(diffs)}]")

    print(f"\n{sep2}\n")


# ── main pipeline function ────────────────────────────────────────

def run_gait_pipeline(
    path:          str,
    label:         str   = "Runner",
    fps:           float = FPS,
    smooth_window: int   = SMOOTH_WINDOW,
    prominence:    float = PROMINENCE,
    min_distance:  int   = MIN_DISTANCE,
    search_back:   int   = SEARCH_BACK,
    verbose:       bool  = True,
    plot:          bool  = False,
) -> dict:
    """
    Full pipeline: load → step finding → strike type → initial contacts.

    Parameters
    ----------
    path           : path to JSONL file
    label          : runner name for display
    fps            : frames per second
    smooth_window  : Savitzky-Golay smoothing window (default 11)
    prominence     : pelvis peak prominence threshold (default 15)
    min_distance   : min frames between pelvis peaks (default 10)
    search_back    : frames to search before midstance for IC (default 10)
    verbose        : print results (default True)
    plot           : show diagnostic plot (default False)

    Returns
    -------
    dict with keys:
        frames        : raw frame list
        peaks         : midstance array indices
        foot_labels   : ["L","R",...] per peak
        strike_type   : "HEEL" / "MID" / "TOE"
        confidence    : "high" / "medium" / "low"
        metrics_agree : bool
        mn_max        : primary strike metric value
        mn_amax       : validation strike metric value (degrees)
        contacts      : list of contact dicts, each with:
                            frame, side, midstance_frame,
                            frames_to_midstance, contact_joint, fallback
    """
    frames = load_jsonl(path)

    # step 1
    peaks, foot_labels = find_midstance_peaks(
        frames,
        smooth_window = smooth_window,
        prominence    = prominence,
        min_distance  = min_distance,
    )

    # step 2
    strike = detect_strike_type(frames)

    # step 3
    contacts = find_initial_contacts(
        frames      = frames,
        peaks       = peaks,
        foot_labels = foot_labels,
        strike_type = strike["overall"],
        search_back = search_back,
    )

    if verbose:
        print_results(path, label, fps, len(frames),
                      peaks, foot_labels, strike, contacts)

    if plot:
        plot_results(frames, peaks, foot_labels, contacts,
                     strike["overall"], label)

    return dict(
        frames        = frames,
        peaks         = peaks,
        foot_labels   = foot_labels,
        strike_type   = strike["overall"],
        confidence    = strike["confidence"],
        metrics_agree = strike["metrics_agree"],
        mn_max        = strike["mn_max"],
        mn_amax       = strike["mn_amax"],
        contacts      = contacts,
    )


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Integrated gait pipeline: step finding + strike type + contact frames"
    )
    parser.add_argument("jsonl_file")
    parser.add_argument("--label",         default="Runner")
    parser.add_argument("--fps",           type=float, default=FPS)
    parser.add_argument("--smooth-window", type=int,   default=SMOOTH_WINDOW)
    parser.add_argument("--prominence",    type=float, default=PROMINENCE)
    parser.add_argument("--min-distance",  type=int,   default=MIN_DISTANCE)
    parser.add_argument("--search-back",   type=int,   default=SEARCH_BACK)
    parser.add_argument("--plot",          action="store_true")
    args = parser.parse_args()

    run_gait_pipeline(
        path          = args.jsonl_file,
        label         = args.label,
        fps           = args.fps,
        smooth_window = args.smooth_window,
        prominence    = args.prominence,
        min_distance  = args.min_distance,
        search_back   = args.search_back,
        plot          = args.plot,
    )


if __name__ == "__main__":
    main()