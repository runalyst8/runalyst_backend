"""
Unified Gait Analysis Pipeline

Orchestrates all analysis modules and consolidates results into:
1. Structured return values from all modules
2. Centralized print output (no scattered prints)
3. PNG plots saved instead of displayed
4. JSON output file with all results
"""

import json
import sys
import os
from typing import Optional
import argparse

# Add current directory to path for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import all analysis modules
from contact_and_overstride import run_gait_pipeline
from new_strike_analysis import analyze_strike_type
from pelvis_analysis import cadence_and_step_vertical_comparison
from swing_stance_analysis import plot_ankle_x as analyze_swing_stance
from trunk_lean_analysis import analyze_forward_trunk_lean
from knee_flexion import plot_ankle_x as analyze_knee_flexion
from overstride_at_contact import calculate_overstride_at_contact


def run_full_pipeline(
    path: str,
    label: str = "Runner",
    fps: float = 64.0,
    output_dir: str = "pipeline_output",
    verbose: bool = True,
) -> dict:
    """
    Run all gait analysis modules and consolidate results.

    Parameters
    ----------
    path : str
        Path to JSONL file with gait data
    label : str
        Label for the runner/subject
    fps : float
        Frames per second (default 64)
    output_dir : str
        Directory to save plots and JSON output
    verbose : bool
        Print results during processing

    Returns
    -------
    dict
        Consolidated results from all modules with keys:
        - contact_and_overstride
        - strike_analysis_new_results
        - strike_analysis_old_results
        - pelvis_analysis_results
        - swing_stance_results
        - trunk_lean_results
        - knee_flexion_results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "metadata": {
            "input_file": path,
            "label": label,
            "fps": fps,
            "output_directory": output_dir,
        },
        "modules": {}
    }

    print("\n" + "=" * 90)
    print("  UNIFIED GAIT ANALYSIS PIPELINE")
    print("=" * 90)
    print(f"\n  Input file: {path}")
    print(f"  Label:      {label}")
    print(f"  FPS:        {fps}")
    print(f"  Output dir: {output_dir}\n")

    # ── MODULE 1: Contact and Overstride (integrated step finding + strike + IC) ───────
    if verbose:
        print("  [1/8] Running contact_and_overstride (step finding + strike + IC)...")
    try:
        gait_result = run_gait_pipeline(
            path=path,
            label=label,
            fps=fps,
            verbose=False,  # Suppress internal prints; we'll collect results
        )
        results["modules"]["contact_and_overstride"] = {
            "status": "success",
            "frames": gait_result.get("frames", []),
            "peaks": gait_result.get("peaks", []),
            "foot_labels": gait_result.get("foot_labels", []),
            "strike_type": gait_result.get("strike_type", "unknown"),
            "confidence": gait_result.get("confidence", "unknown"),
            "contacts": gait_result.get("contacts", []),
            "overstride": gait_result.get("overstride", {}),
        }
        if verbose:
            print("    ✓ Contact and Overstride complete")
    except Exception as e:
        results["modules"]["contact_and_overstride"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ Contact and Overstride error: {e}")

    # ── MODULE 2: New Strike Analysis (detailed metrics) ─────────────────────
    if verbose:
        print("  [2/7] Running new_strike_analysis...")
    try:
        strike_new_result = analyze_strike_type(
            path=path,
            label=label,
            fps=fps,
            plot=False,  # Don't plot from the module itself
        )
        results["modules"]["strike_analysis_new"] = strike_new_result
        if verbose:
            print("    ✓ New strike analysis complete")
    except Exception as e:
        results["modules"]["strike_analysis_new"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ New strike analysis error: {e}")

    # ── MODULE 3: Pelvis Analysis (cadence & excursion) ──────────────────────
    if verbose:
        print("  [3/7] Running pelvis_analysis (cadence & excursion)...")
    try:
        pelvis_plot_path = os.path.join(output_dir, f"{label}_pelvis_analysis.png")
        pelvis_result = cadence_and_step_vertical_comparison(
            path=path,
            label=label,
            smooth_window=11,
            prominence=15.0,
            min_distance=10,
            fps=fps,
            save_path=pelvis_plot_path,
        )
        results["modules"]["pelvis_analysis"] = pelvis_result
        if verbose:
            print("    ✓ Pelvis analysis complete")
    except Exception as e:
        results["modules"]["pelvis_analysis"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ Pelvis analysis error: {e}")

    # ── MODULE 4: Swing/Stance Analysis (flight phase metrics) ───────────────
    if verbose:
        print("  [4/7] Running swing_stance_analysis (flight metrics)...")
    try:
        swing_plot_path = os.path.join(output_dir, f"{label}_swing_stance.png")
        swing_result = analyze_swing_stance(path, save_path=swing_plot_path)
        # Remove any matplotlib figures from the result to keep JSON serializable
        if "flight_metrics" in swing_result:
            swing_summary = {
                "flight_metrics": swing_result.get("flight_metrics", {}),
                "left_foot_cycles": swing_result.get("left_foot_cycles", [])
                    if hasattr(swing_result, "left_foot_cycles") else None,
                "right_foot_cycles": swing_result.get("right_foot_cycles", [])
                    if hasattr(swing_result, "right_foot_cycles") else None,
                "plot_path": swing_plot_path,
            }
            results["modules"]["swing_stance_analysis"] = swing_summary
        else:
            results["modules"]["swing_stance_analysis"] = swing_result
        if verbose:
            print("    ✓ Swing/stance analysis complete")
    except Exception as e:
        results["modules"]["swing_stance_analysis"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ Swing/stance analysis error: {e}")

    # ── MODULE 5: Trunk Lean Analysis ──────────────────────────────────────────
    if verbose:
        print("  [5/7] Running trunk_lean_analysis...")
    try:
        trunk_result = analyze_forward_trunk_lean(
            path=path,
            label=label,
            fps=fps,
            plot=False,  # Don't plot from the module itself
        )
        # Convert numpy arrays to lists for JSON serialization
        trunk_summary = {
            "mean_global": float(trunk_result.get("mean_global", 0)),
            "std_global": float(trunk_result.get("std_global", 0)),
            "min_global": float(trunk_result.get("min_global", 0)),
            "max_global": float(trunk_result.get("max_global", 0)),
            "mean_lower": float(trunk_result.get("mean_lower", 0)),
            "std_lower": float(trunk_result.get("std_lower", 0)),
            "min_lower": float(trunk_result.get("min_lower", 0)),
            "max_lower": float(trunk_result.get("max_lower", 0)),
            "mean_upper": float(trunk_result.get("mean_upper", 0)),
            "std_upper": float(trunk_result.get("std_upper", 0)),
            "min_upper": float(trunk_result.get("min_upper", 0)),
            "max_upper": float(trunk_result.get("max_upper", 0)),
        }
        results["modules"]["trunk_lean_analysis"] = trunk_summary
        if verbose:
            print("    ✓ Trunk lean analysis complete")
    except Exception as e:
        results["modules"]["trunk_lean_analysis"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ Trunk lean analysis error: {e}")

    # ── MODULE 6: Knee Flexion Analysis ────────────────────────────────────────
    if verbose:
        print("  [6/7] Running knee_flexion_analysis...")
    try:
        knee_plot_path = os.path.join(output_dir, f"{label}_knee_flexion.png")
        knee_result = analyze_knee_flexion(path, save_path=knee_plot_path)
        # Extract relevant metrics
        knee_summary = {
            "left_events": knee_result.get("left_events", {}),
            "right_events": knee_result.get("right_events", {}),
            "left_knee_angles": [float(x) for x in knee_result.get("left_knee_angles", [])]
                if "left_knee_angles" in knee_result else None,
            "right_knee_angles": [float(x) for x in knee_result.get("right_knee_angles", [])]
                if "right_knee_angles" in knee_result else None,
            "plot_path": knee_plot_path,
        }
        results["modules"]["knee_flexion_analysis"] = knee_summary
        if verbose:
            print("    ✓ Knee flexion analysis complete")
    except Exception as e:
        results["modules"]["knee_flexion_analysis"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ Knee flexion analysis error: {e}")

    # ── MODULE 7: Alpers Overstride Analysis ────────────────────────────────────
    if verbose:
        print("  [7/7] Running alpers_overstride_analysis...")
    try:
        # Get frames and contacts from contact_and_overstride
        gait_result = results["modules"].get("contact_and_overstride", {})
        if gait_result.get("status") == "success":
            frames = gait_result.get("frames", [])
            contacts = gait_result.get("contacts", [])
            if frames and contacts:
                alpers_result = calculate_overstride_at_contact(frames, contacts, k=0.7)
                results["modules"]["alpers_overstride"] = alpers_result
                if verbose:
                    print("    ✓ Alpers overstride analysis complete")
            else:
                results["modules"]["alpers_overstride"] = {"status": "error", "error": "No frames or contacts from contact_and_overstride"}
                if verbose:
                    print("    ✗ Alpers overstride analysis error: No frames or contacts from contact_and_overstride")
        else:
            results["modules"]["alpers_overstride"] = {"status": "error", "error": "Contact and Overstride failed"}
            if verbose:
                print("    ✗ Alpers overstride analysis error: Contact and Overstride failed")
    except Exception as e:
        results["modules"]["alpers_overstride"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    ✗ Alpers overstride analysis error: {e}")

    # ── Print consolidated summary ─────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 90)
        print("  PIPELINE SUMMARY")
        print("─" * 90)

        # Contact and Overstride Summary
        gp = results["modules"].get("contact_and_overstride", {})
        if gp.get("status") == "success":
            print(f"\n  ▸ CONTACT AND OVERSTRIDE")
            print(f"    Strike Type:       {gp.get('strike_type', '?')}")
            print(f"    Confidence:        {gp.get('confidence', '?')}")
            peaks = gp.get('peaks', [])
            peak_list = [int(x) for x in peaks] if hasattr(peaks, '__iter__') else [int(peaks)]
            print(f"    Midstance frames:  {len(peak_list)} detected")
            print(f"    Midstance indices: {peak_list}")
            contacts = gp.get('contacts', [])
            contact_frames = [int(c.get('frame')) for c in contacts if isinstance(c, dict) and 'frame' in c]
            print(f"    Contacts detected: {len(contact_frames)} frames")
            print(f"    Contact frames:    {contact_frames}")
            ov = gp.get('overstride', {})
            if isinstance(ov, tuple) and len(ov) >= 4:
                print(f"    Overstride:        {ov[2]} / {ov[3]} steps (mean: {ov[0]:+.1f})")

        # Strike Analysis Summary
        san = results["modules"].get("strike_analysis_new", {})
        if "status" not in san:
            print(f"\n  ▸ STRIKE ANALYSIS")
            print(f"    Primary:           {san.get('primary', '?')}")
            print(f"    Validation:        {san.get('validation', '?')}")
            print(f"    Confidence:        {san.get('confidence', '?')}")

        # Pelvis Summary
        pa = results["modules"].get("pelvis_analysis", {})
        if "status" not in pa:
            print(f"\n  ▸ PELVIS ANALYSIS")
            print(f"    Cadence:           {pa.get('cadence_steps_per_min', 0):.1f} steps/min")
            summary = pa.get('summary', {})
            if summary:
                print(f"    Excursion L:       {summary.get('avg_excursion_L', 0):.2f}")
                print(f"    Excursion R:       {summary.get('avg_excursion_R', 0):.2f}")

        # Trunk Lean Summary
        ta = results["modules"].get("trunk_lean_analysis", {})
        if "status" not in ta:
            print(f"\n  ▸ TRUNK LEAN ANALYSIS")
            print(f"    Global angle:      {ta.get('mean_global', 0):+.2f}°")
            print(f"    Lower (pelvis):    {ta.get('mean_lower', 0):+.2f}°")
            print(f"    Upper (thoracic):  {ta.get('mean_upper', 0):+.2f}°")

        # Swing/Stance Summary
        ssa = results["modules"].get("swing_stance_analysis", {})
        if "status" not in ssa:
            print(f"\n  ▸ SWING/STANCE ANALYSIS")
            flight = ssa.get('flight_metrics', {})
            if flight and 'overall_averages' in flight:
                ov_avg = flight['overall_averages']
                print(f"    Left flight:       {ov_avg.get('avg_left_flight', 0):.1f}%")
                print(f"    Right flight:      {ov_avg.get('avg_right_flight', 0):.1f}%")
                print(f"    Double flight:     {ov_avg.get('avg_double_flight', 0):.1f}%")

        # Knee Flexion Summary
        ka = results["modules"].get("knee_flexion_analysis", {})
        if "status" not in ka:
            print(f"\n  ▸ KNEE FLEXION ANALYSIS")
            l_events = ka.get('left_events', {})
            r_events = ka.get('right_events', {})
            print(f"    Left foot cycles:  {len(l_events.get('foot_strike', []))} detected")
            print(f"    Right foot cycles: {len(r_events.get('foot_strike', []))} detected")

        # Alpers Overstride Summary
        ao = results["modules"].get("alpers_overstride", {})
        if "status" not in ao:
            print(f"\n  ▸ ALPERS OVERSTRIDE")
            mean_osi = ao.get('mean_overstride_index_deg')
            if mean_osi is not None:
                print(f"    Mean overstride index: {mean_osi:.2f}°")
                print(f"    Mean alpha:           {ao.get('mean_alpha_deg', 0):.2f}°")
                print(f"    Mean lean forward:    {ao.get('mean_lean_forward_deg', 0):.2f}°")
                print(f"    Comment:              {ao.get('comment', 'N/A')}")
            else:
                print(f"    No overstride data available")

        print("\n" + "=" * 90 + "\n")

    # ── Save JSON output ───────────────────────────────────────────────────────
    json_output_path = os.path.join(output_dir, f"{label}_results.json")
    with open(json_output_path, "w") as f:
        # Make results JSON serializable
        json_results = {
            "metadata": results["metadata"],
            "modules": {}
        }
        for module_name, module_data in results["modules"].items():
            json_results["modules"][module_name] = _make_serializable(module_data)

        json.dump(json_results, f, indent=2, default=str)

    if verbose:
        print(f"  Results saved to: {json_output_path}\n")

    results["json_output_path"] = json_output_path
    return results


def _make_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-compatible formats."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Gait Analysis Pipeline"
    )
    parser.add_argument("jsonl_file", help="Path to JSONL file with gait data")
    parser.add_argument("--label", default="Runner", help="Subject label (default: Runner)")
    parser.add_argument("--fps", type=float, default=64.0, help="Frames per second (default: 64)")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    results = run_full_pipeline(
        path=args.jsonl_file,
        label=args.label,
        fps=args.fps,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    return results


if __name__ == "__main__":
    main()
