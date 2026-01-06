import numpy as np
import matplotlib
matplotlib.use("Agg")

from contact import (
    load_ok_rows,
    get_feet,
    detect_contacts_root_relative,
    stance_segments,
)

from temporal_metrics import (
    stance_to_step_times,
    same_foot_step_times,
    alternating_step_times,
    cadence,
    symmetry_index,
)


def main():
    rows = load_ok_rows("mid_strike.jsonl")
    t = np.array([r["timestamp_sec"] for r in rows])

    L, R = [], []
    for r in rows:
        l, r_ = get_feet(r["nlf"])
        L.append(l)
        R.append(r_)

    L = np.vstack(L)
    R = np.vstack(R)

    # auto-detect vertical axis (smallest motion)
    ranges = L.max(axis=0) - L.min(axis=0)
    vertical_axis = np.argmin(ranges)

    print("Detected vertical axis:", vertical_axis)

    # contact detection (ROOT-RELATIVE SAFE)
    cL = detect_contacts_root_relative(L, t, vertical_axis)
    cR = detect_contacts_root_relative(R, t, vertical_axis)

    segL = stance_segments(cL, t)
    segR = stance_segments(cR, t)

    print("Left stance segments:", len(segL))
    print("Right stance segments:", len(segR))

    # temporal metrics
    tL = stance_to_step_times(segL)
    tR = stance_to_step_times(segR)

    L_steps = same_foot_step_times(tL)
    R_steps = same_foot_step_times(tR)
    ALT_steps = alternating_step_times(tL, tR)

    print("Alternating steps:", len(ALT_steps))
    print("Cadence (spm):", cadence(ALT_steps))
    print("Step time symmetry (%):", symmetry_index(L_steps, R_steps))


if __name__ == "__main__":
    main()
