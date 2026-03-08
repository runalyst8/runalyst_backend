
import os
import glob
import argparse
import numpy as np

from io_utility import open_nlf_output_json, extract_ys_and_times
from math_utility import get_dt, smooth, find_y_peaks


def detect_strike_for_side(video_nlf, side, threshold_sec=0.03):
    heel_name = f"{side}_heel"
    toe_name = f"{side}_toe"

    try:
        times_h, ys_h = extract_ys_and_times(video_nlf, "nlf.joints3d[0]", heel_name)
        times_t, ys_t = extract_ys_and_times(video_nlf, "nlf.joints3d[0]", toe_name)
    except SystemExit:
        return "unknown", None, None

    if len(times_h) == 0 or len(times_t) == 0:
        return "unknown", None, None

    
    # Smooth signals
    dts_h = get_dt(times_h)
    dts_t = get_dt(times_t)
    dts = float(np.median([dts_h, dts_t]))

    ys_h_s = smooth(ys_h, dts)
    ys_t_s = smooth(ys_t, dts)


    # Find maxima (peaks) — in these NLF outputs contact corresponds to peaks
    peaks_h = find_y_peaks(ys_h_s, dts)
    peaks_t = find_y_peaks(ys_t_s, dts)

    def select_first_valid_peak(peaks, ys_s, times, dts):
        if peaks is None:
            peaks = np.array([], dtype=int)
        peaks = np.array(peaks, dtype=int)

        # ignore very early/late frames which can be noisy/start-stop artifacts
        margin = max(0.05, 3 * dts)
        t_start = times[0] + margin
        t_end = times[-1] - margin

        # candidate peaks within time window
        cand = [int(p) for p in peaks if (times[int(p)] >= t_start and times[int(p)] <= t_end)]

        # amplitude threshold: require peak height > median + 0.5*std
        if len(ys_s) > 0:
            amp_thr = float(np.median(ys_s) + 0.5 * np.std(ys_s))
        else:
            amp_thr = -np.inf

        for idx in cand:
            if ys_s[int(idx)] >= amp_thr:
                return int(idx)

        # fallback: first candidate regardless of amplitude
        if len(cand) > 0:
            return int(cand[0])

        # fallback to global max if it is not at edges
        if len(ys_s) > 0:
            imax = int(np.argmax(ys_s))
            if times[imax] >= t_start and times[imax] <= t_end:
                return imax

        # last resort: earliest peak even if near edges
        if len(peaks) > 0:
            return int(peaks[0])

        return None

    sel_h = select_first_valid_peak(peaks_h, ys_h_s, times_h, dts)
    sel_t = select_first_valid_peak(peaks_t, ys_t_s, times_t, dts)

    if sel_h is not None:
        t_h = float(times_h[int(sel_h)])
    else:
        t_h = None

    if sel_t is not None:
        t_t = float(times_t[int(sel_t)])
    else:
        t_t = None

    # if either is None, try safe global fallbacks
    if t_h is None and len(ys_h_s) > 0:
        t_h = float(times_h[int(np.argmax(ys_h_s))])
    if t_t is None and len(ys_t_s) > 0:
        t_t = float(times_t[int(np.argmax(ys_t_s))])

    dt = abs(t_h - t_t)
    if dt <= threshold_sec:
        return "simultaneous", t_h, t_t
    if t_h < t_t:
        return "heel", t_h, t_t
    return "toe", t_h, t_t


def analyze_file(jsonl_path, threshold_sec=0.03):
    video_nlf = open_nlf_output_json(jsonl_path)
    results = {}
    for side in ("left", "right"):
        typ, t_h, t_t = detect_strike_for_side(video_nlf, side, threshold_sec=threshold_sec)
        results[side] = {"type": typ, "heel_time": t_h, "toe_time": t_t}
        #print(f"{jsonl_path} - {side} foot strike: {typ} (heel_time={t_h}, toe_time={t_t})")
    return results


def main(folder="foot_strike_examples_jsons", threshold=0.03):
    pattern = os.path.join(folder, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No JSONL files found in", folder)
        return

    for p in files:
        try:
            res = analyze_file(p, threshold_sec=threshold)
        except Exception as e:
            print(p, "=> error:", e)
            continue
        print(p, "=> left:", res["left"]["type"], ", right:", res["right"]["type"]) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze foot strike type in JSONL outputs")
    parser.add_argument("--folder", default="foot_strike_examples_jsons", help="Folder with jsonl files")
    parser.add_argument("--threshold", type=float, default=0.03, help="Time threshold for simultaneous (seconds)")
    args = parser.parse_args()
    main(folder=args.folder, threshold=args.threshold)
