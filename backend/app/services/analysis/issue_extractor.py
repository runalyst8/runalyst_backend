from __future__ import annotations

from typing import Any


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_issues(modules: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw analysis modules into a labelled list of issues with severity."""
    issues: list[dict[str, Any]] = []

    # ── Cadence ──────────────────────────────────────────────────────────────
    pelvis = modules.get("pelvis_analysis") or {}
    cadence = _num(pelvis.get("cadence_steps_per_min"))
    if cadence is not None:
        if cadence < 160:
            issues.append({
                "issue_key": "low_cadence",
                "name": "Low Cadence",
                "severity": "high",
                "measured_value": f"{cadence:.0f} spm",
                "threshold": "< 160 spm",
                "context": f"Cadence is {cadence:.0f} steps/min. Optimal range is 170–180 spm.",
            })
        elif cadence < 170:
            issues.append({
                "issue_key": "low_cadence",
                "name": "Low Cadence",
                "severity": "moderate",
                "measured_value": f"{cadence:.0f} spm",
                "threshold": "< 170 spm",
                "context": f"Cadence is {cadence:.0f} steps/min. Optimal range is 170–180 spm.",
            })

    # ── Overstride Index ─────────────────────────────────────────────────────
    overstride2 = modules.get("overstride_metric_2") or {}
    oi = _num(overstride2.get("mean_overstride_index_deg"))
    if oi is not None:
        if oi >= 15:
            issues.append({
                "issue_key": "overstride",
                "name": "Overstride",
                "severity": "high",
                "measured_value": f"{oi:.1f}°",
                "threshold": ">= 15°",
                "context": f"Overstride index is {oi:.1f}°. Acceptable < 5°, mild 5–10°, overstride >= 10°.",
            })
        elif oi >= 10:
            issues.append({
                "issue_key": "overstride",
                "name": "Overstride",
                "severity": "moderate",
                "measured_value": f"{oi:.1f}°",
                "threshold": ">= 10°",
                "context": f"Overstride index is {oi:.1f}°. Acceptable < 5°, mild 5–10°, overstride >= 10°.",
            })
        elif oi >= 5:
            issues.append({
                "issue_key": "overstride",
                "name": "Overstride",
                "severity": "mild",
                "measured_value": f"{oi:.1f}°",
                "threshold": ">= 5°",
                "context": f"Overstride index is {oi:.1f}°. Acceptable < 5°, mild 5–10°, overstride >= 10°.",
            })

    # ── Strike Pattern ───────────────────────────────────────────────────────
    strike = modules.get("strike_analysis_new") or {}
    strike_type = str(strike.get("overall") or "").upper()
    strike_conf = str(strike.get("confidence") or "").lower()
    if strike_type == "HEEL":
        severity_map = {"high": "high", "medium": "moderate", "low": "mild"}
        issues.append({
            "issue_key": "heel_strike",
            "name": "Heel Strike",
            "severity": severity_map.get(strike_conf, "moderate"),
            "measured_value": f"HEEL ({strike_conf} confidence)",
            "threshold": "MIDFOOT or FOREFOOT preferred",
            "context": "Heel striking increases braking impulse and loads the knee and shin.",
        })

    # ── Trunk Lean ───────────────────────────────────────────────────────────
    trunk = modules.get("trunk_lean_analysis") or {}
    mean_global = _num(trunk.get("mean_global"))
    std_global = _num(trunk.get("std_global"))
    if mean_global is not None:
        if mean_global > 20:
            issues.append({
                "issue_key": "excessive_trunk_lean",
                "name": "Excessive Forward Trunk Lean",
                "severity": "high",
                "measured_value": f"{mean_global:.1f}°",
                "threshold": "> 20°",
                "context": f"Mean forward trunk lean is {mean_global:.1f}°. Values above 15° may indicate fatigue or poor posture.",
            })
        elif mean_global > 15:
            issues.append({
                "issue_key": "excessive_trunk_lean",
                "name": "Excessive Forward Trunk Lean",
                "severity": "moderate",
                "measured_value": f"{mean_global:.1f}°",
                "threshold": "> 15°",
                "context": f"Mean forward trunk lean is {mean_global:.1f}°. Values above 15° may indicate fatigue or poor posture.",
            })
    if std_global is not None and std_global > 6:
        issues.append({
            "issue_key": "unstable_trunk",
            "name": "Unstable Trunk",
            "severity": "moderate",
            "measured_value": f"std {std_global:.1f}°",
            "threshold": "std > 6°",
            "context": f"Trunk lean variability (std {std_global:.1f}°) is high, suggesting core instability.",
        })

    # ── Stride Asymmetry ─────────────────────────────────────────────────────
    stride_l = _num(pelvis.get("mean_stride_L"))
    stride_r = _num(pelvis.get("mean_stride_R"))
    if stride_l is not None and stride_r is not None:
        mean_stride = (stride_l + stride_r) / 2
        if mean_stride > 0:
            asym = abs(stride_l - stride_r) / mean_stride
            if asym > 0.10:
                issues.append({
                    "issue_key": "stride_asymmetry",
                    "name": "Stride Length Asymmetry",
                    "severity": "moderate",
                    "measured_value": f"{asym * 100:.1f}% difference",
                    "threshold": "> 10% difference",
                    "context": (
                        f"Left stride {stride_l:.1f} px vs right {stride_r:.1f} px "
                        f"({asym * 100:.1f}% asymmetry). May indicate muscle imbalance."
                    ),
                })

    # ── Pelvic Vertical Asymmetry ─────────────────────────────────────────────
    summary = pelvis.get("summary") or {}
    exc_l = _num(summary.get("avg_excursion_L"))
    exc_r = _num(summary.get("avg_excursion_R"))
    if exc_l is not None and exc_r is not None:
        mean_exc = (exc_l + exc_r) / 2
        if mean_exc > 0:
            pelvic_asym = abs(exc_l - exc_r) / mean_exc
            if pelvic_asym > 0.15:
                issues.append({
                    "issue_key": "pelvic_asymmetry",
                    "name": "Pelvic Vertical Asymmetry",
                    "severity": "moderate",
                    "measured_value": f"{pelvic_asym * 100:.1f}% difference",
                    "threshold": "> 15% difference",
                    "context": (
                        f"Left pelvic excursion {exc_l:.2f} vs right {exc_r:.2f} "
                        f"({pelvic_asym * 100:.1f}% asymmetry). May indicate hip weakness."
                    ),
                })

    issues.sort(key=lambda i: {"high": 0, "moderate": 1, "mild": 2}.get(i["severity"], 3))
    return issues
