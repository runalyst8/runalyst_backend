"""
Strike type detection — normalized ankle/foot Y difference method.

Algorithm
---------
Primary metric:
    norm_diff  = (ankle_y - foot_y) / shin_length          [dimensionless]
    shin_length = median 3D distance(knee, ankle)           [normalization]

    mn_max = min( max(L_norm_diff), max(R_norm_diff) )

    The maximum of norm_diff across all frames captures the moment when
    ankle is lowest relative to the foot — i.e. heel-to-toe orientation
    at initial contact. Taking the minimum of both sides requires both
    feet to agree, suppressing single-side noise.

    mn_max > -0.038          →  Heel Strike
    -0.096 < mn_max ≤ -0.038 →  Mid Strike
    mn_max ≤ -0.096          →  Toe Strike

Validation metric:
    angle = atan2(ankle_y - foot_y, horizontal_dist)    [degrees]
    mn_amax = min( max(L_angle), max(R_angle) )

    mn_amax > -6.5°           →  Heel
    -16.3° < mn_amax ≤ -6.5°  →  Mid
    mn_amax ≤ -16.3°           →  Toe

    Thresholds calibrated on 32 files at 90.6% accuracy.

Key difference from foot-angle window method (strike_detection.py):
    This method does NOT detect contact frames. It scans all frames and
    uses the global maximum as a proxy for initial contact orientation.
    More robust to frame detection errors; less informative per-step.

Limitations:
    - SMPL ankle != heel bone, foot != toe tip
    - Both metrics fail on the same 3 files (overstride toe-labelled
      runners with heel-like mechanics)
    - Calibrated on fixed camera setup; revalidate for new angles

SMPL joints used:
    L_Knee[4], R_Knee[5], L_Ankle[7], R_Ankle[8], L_Foot[10], R_Foot[11]

Usage:
    python strike_analysis.py your_file.jsonl
    python strike_analysis.py your_file.jsonl --label "Alper" --fps 64

    # import in another script:
    from strike_analysis import analyze_strike_type
    result = analyze_strike_type(path="file.jsonl", label="Alper", plot=False)
    print(result["overall"])           # "HEEL" / "MID" / "TOE" / "UNCLEAR"
    print(result["mean_norm_diff"])    # primary metric value
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from typing import Optional

FPS = 64.0

# ── SMPL joint indices ────────────────────────────────────────────
L_KNEE  = 4;  R_KNEE  = 5
L_ANKLE = 7;  R_ANKLE = 8
L_FOOT  = 10; R_FOOT  = 11

# ── Thresholds ────────────────────────────────────────────────────
TH_HEEL   = -0.038   # primary: norm_diff
TH_TOE    = -0.096
TH_HEEL_A = -6.5     # validation: angle (degrees)
TH_TOE_A  = -16.3


# ── helpers ───────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    frames.sort(key=lambda x: x["frame_index"])
    return frames


def get_joint(frames: list[dict], idx: int) -> np.ndarray:
    return np.array([f["joints_3d"][0][0][idx] for f in frames])


def _classify_norm(mn_max: float) -> str:
    if mn_max > TH_HEEL:  return "HEEL"
    if mn_max > TH_TOE:   return "MID"
    return "TOE"


def _classify_angle(mn_amax: float) -> str:
    if mn_amax > TH_HEEL_A:  return "HEEL"
    if mn_amax > TH_TOE_A:   return "MID"
    return "TOE"


def _confidence(mn_max: float, mn_amax: float) -> str:
    primary   = _classify_norm(mn_max)
    secondary = _classify_angle(mn_amax)
    margin    = min(abs(mn_max - TH_HEEL), abs(mn_max - TH_TOE))
    if primary == secondary and margin > 0.025:
        return "high"
    elif primary == secondary:
        return "medium"
    elif margin > 0.02:
        return "medium"
    return "low"


# ── per-side computation ──────────────────────────────────────────

def _analyze_side(frames: list[dict],
                  knee_idx: int,
                  ankle_idx: int,
                  foot_idx: int) -> dict:
    """
    Compute norm_diff and angle arrays for one side (L or R).

    norm_diff = (ankle_y - foot_y) / shin_length
        positive → ankle below foot → heel orientation
        negative → foot below ankle → toe orientation

    angle = atan2(ankle_y - foot_y, horizontal_dist)
        positive → heel orientation
        negative → toe orientation
    """
    knee  = get_joint(frames, knee_idx)
    ankle = get_joint(frames, ankle_idx)
    foot  = get_joint(frames, foot_idx)

    shin_len = float(np.median(
        np.sqrt(np.sum((knee - ankle) ** 2, axis=1))
    ))

    diff      = ankle[:, 1] - foot[:, 1]
    norm_diff = diff / shin_len

    d   = ankle - foot
    dxz = np.clip(np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2), 1e-6, None)
    angles = np.degrees(np.arctan2(d[:, 1], dxz))

    return dict(
        ankle_y   = ankle[:, 1],
        foot_y    = foot[:, 1],
        norm_diff = norm_diff,
        angles    = angles,
        shin_len  = shin_len,
        max_nd    = float(np.max(norm_diff)),
        max_angle = float(np.max(angles)),
    )


# ── core computation ──────────────────────────────────────────────

def compute_strike_metrics(frames: list[dict]) -> dict:
    """
    Compute primary (norm_diff) and validation (angle) metrics
    for both feet.

    Returns
    -------
    dict with keys:
        left, right          — per-side result dicts
        mn_max               — min(max_L_norm_diff, max_R_norm_diff)
        mn_amax              — min(max_L_angle, max_R_angle)
        primary              — "HEEL" / "MID" / "TOE"   (norm_diff)
        validation           — "HEEL" / "MID" / "TOE"   (angle)
        confidence           — "high" / "medium" / "low"
        metrics_agree        — bool
    """
    left  = _analyze_side(frames, L_KNEE, L_ANKLE, L_FOOT)
    right = _analyze_side(frames, R_KNEE, R_ANKLE, R_FOOT)

    mn_max  = min(left["max_nd"],    right["max_nd"])
    mn_amax = min(left["max_angle"], right["max_angle"])

    primary    = _classify_norm(mn_max)
    validation = _classify_angle(mn_amax)
    conf       = _confidence(mn_max, mn_amax)

    return dict(
        left          = left,
        right         = right,
        mn_max        = mn_max,
        mn_amax       = mn_amax,
        primary       = primary,
        validation    = validation,
        confidence    = conf,
        metrics_agree = (primary == validation),
    )


# ── printing ──────────────────────────────────────────────────────

def build_strike_summary(metrics: dict, path: str, label: str,
                         fps: float, n_frames: int) -> dict:
    return {
        'path': path,
        'label': label,
        'fps': fps,
        'n_frames': n_frames,
        'primary': metrics['primary'],
        'validation': metrics['validation'],
        'confidence': metrics['confidence'],
        'mn_max': metrics['mn_max'],
        'mn_amax': metrics['mn_amax'],
        'metrics_agree': metrics['metrics_agree'],
        'left': metrics['left'],
        'right': metrics['right'],
    }


# ── plotting ──────────────────────────────────────────────────────

def plot_results(metrics: dict, path: str, label: str,
                 save_path: Optional[str] = None) -> Optional[str]:
    L   = metrics["left"]
    R   = metrics["right"]
    fn  = os.path.basename(path)
    n   = len(L["ankle_y"])
    t   = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor('#04080f')
    for ax in axes.flat:
        ax.set_facecolor('#080f1c')
        ax.tick_params(labelcolor='#64748b', labelsize=8)
        ax.spines[['top','right','bottom','left']].set_color('#1e3554')
        ax.grid(color='#1e3554', linewidth=0.4, linestyle=':')

    agree_str = "metrics agree" if metrics["metrics_agree"] else "metrics disagree"
    fig.suptitle(
        f"Strike detection  —  {label}  |  {fn}\n"
        f"Result: {metrics['primary']}  "
        f"({metrics['confidence']} confidence, {agree_str})   "
        f"mn_max={metrics['mn_max']:+.4f}  mn_amax={metrics['mn_amax']:+.1f}°",
        color='#e2e8f0', fontsize=11, fontweight='bold',
    )

    for col, (side_data, side_label) in enumerate([(L, "Left"), (R, "Right")]):
        C_ANKLE = '#f97316'
        C_FOOT  = '#4ade80'
        C_ND    = '#a78bfa'

        # top: ankle Y vs foot Y
        ax = axes[0, col]
        ax.plot(t, side_data["ankle_y"], color=C_ANKLE, lw=1.4,
                label="ankle Y")
        ax.plot(t, side_data["foot_y"],  color=C_FOOT,  lw=1.4,
                label="foot Y")
        ax.set_title(f"{side_label} foot — raw Y", color='#94a3b8', fontsize=9)
        ax.set_ylabel("Y (down = +)", color='#94a3b8', fontsize=8)
        ax.legend(fontsize=8, facecolor='#0c1526',
                  edgecolor='#1e3554', labelcolor='#94a3b8')

        # bottom: norm_diff with thresholds
        ax = axes[1, col]
        ax.plot(t, side_data["norm_diff"], color=C_ND, lw=1.2, alpha=0.7,
                label="norm_diff")
        ax.axhline(side_data["max_nd"], color='#f97316', lw=1.5,
                   ls='--', label=f"max = {side_data['max_nd']:+.4f}")
        ax.axhline(TH_HEEL, color='#fbbf24', lw=1.0, ls=':',
                   label=f"HEEL/MID ({TH_HEEL})")
        ax.axhline(TH_TOE,  color='#f472b6', lw=1.0, ls=':',
                   label=f"MID/TOE ({TH_TOE})")
        ax.axhline(0, color='#334155', lw=0.8)
        ax.fill_between(t, side_data["norm_diff"], TH_HEEL,
                        where=side_data["norm_diff"] > TH_HEEL,
                        alpha=0.1, color='#f97316')
        ax.fill_between(t, side_data["norm_diff"], TH_TOE,
                        where=side_data["norm_diff"] < TH_TOE,
                        alpha=0.1, color='#4ade80')
        ax.set_title(
            f"{side_label}: (ankle_y − foot_y) / shin  →  "
            f"{_classify_norm(side_data['max_nd'])}",
            color='#94a3b8', fontsize=9,
        )
        ax.set_ylabel("norm_diff (dimensionless)", color='#94a3b8', fontsize=8)
        ax.set_xlabel("Frame", color='#64748b', fontsize=8)
        ax.legend(fontsize=8, facecolor='#0c1526',
                  edgecolor='#1e3554', labelcolor='#94a3b8')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


# ── main analysis function ────────────────────────────────────────

def analyze_strike_type(
    path:       str,
    label:      str   = "Runner",
    fps:        float = FPS,
    plot:       bool  = True,
) -> dict:
    """
    Full pipeline: load -> compute metrics -> print summary -> (optionally) plot.

    Parameters
    ----------
    path    : path to JSONL file
    label   : runner name for display
    fps     : frames per second (used only for duration display)
    plot    : whether to show the diagnostic plot

    Returns
    -------
    dict with keys:
        overall          — "HEEL" / "MID" / "TOE"  (primary metric result)
        confidence       — "high" / "medium" / "low"
        metrics_agree    — bool (primary and validation agree)
        mn_max           — primary metric value  (dimensionless)
        mn_amax          — validation metric value (degrees)
        primary          — same as overall
        validation       — validation metric result
        left, right      — per-side detail dicts (norm_diff array, angles, etc.)
    """
    frames  = load_jsonl(path)
    metrics = compute_strike_metrics(frames)
    summary = build_strike_summary(metrics, path, label, fps, len(frames))

    if plot:
        save_name = f"{label.replace(' ', '_')}_strike_detection.png"
        plot_results(metrics, path, label, save_path=save_name)
        summary['plot_path'] = save_name

    return dict(
        overall       = metrics["primary"],
        confidence    = metrics["confidence"],
        metrics_agree = metrics["metrics_agree"],
        mn_max        = metrics["mn_max"],
        mn_amax       = metrics["mn_amax"],
        primary       = metrics["primary"],
        validation    = metrics["validation"],
        left          = metrics["left"],
        right         = metrics["right"],
    )


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Strike type detection — normalized ankle/foot Y method"
    )
    parser.add_argument("jsonl_file")
    parser.add_argument("--label", default="Runner")
    parser.add_argument("--fps",   type=float, default=FPS)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    analyze_strike_type(
        path  = args.jsonl_file,
        label = args.label,
        fps   = args.fps,
        plot  = not args.no_plot,
    )


if __name__ == "__main__":
    main()