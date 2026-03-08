import matplotlib.pyplot as plt
from math_utility import *
from io_utility import *

JSONL_PATH = "run_nlf.jsonl"

# İstediğin kaynak:
TARGET_PATH = "nlf.joints3d[0]"          # <-- bunu kullan
# TARGET_PATH = "nlf.joints3d_nonparam[0]"  # alternatif


JOINT_IDX = 8   # sağ ayak için deneyeceğin index (0-23 arası)

video_nlf = open_nlf_output_json(json_path=JSONL_PATH)


times, ys = extract_ys_and_times(video_nlf=video_nlf, target_path=TARGET_PATH, joint_idx=JOINT_IDX)
dts = get_dt(times)
ys_smooth = smooth(ys, dts)
peaks= find_y_peaks(ys_smooth=ys_smooth, dts=dts)


times_heel, ys_heel = extract_ys_and_times(video_nlf=video_nlf, target_path=TARGET_PATH, joint_idx="right_heel")
dts_heel = get_dt(times_heel)
ys_heel_smooth = smooth(ys_heel, dts_heel)
minimums_heel = find_minima_indices(ys_heel_smooth, dts_heel)

times_toe, ys_toe = extract_ys_and_times(video_nlf=video_nlf, target_path=TARGET_PATH, joint_idx="right_toe")
dts_toe = get_dt(times_toe)
ys_toe_smooth = smooth(ys_toe, dts_toe)
minimums_toe = find_minima_indices(ys_toe_smooth, dts_toe)

"""
average_foot_contact_times = (times_heel[minimums_heel] + times_toe[minimums_toe]) / 2

cycles = [(average_foot_contact_times[k], average_foot_contact_times[k+1]) for k in range(len(average_foot_contact_times)-1)]
print(f"Total cycles (using heel & toe minima): {len(cycles)}")
"""

"""
#print(f"Using TARGET_PATH: {TARGET_PATH}  (J={arr.shape[0]} from last valid frame)")
print(f"Valid frames: {len(ys)}")
print(f"Detected contact candidates: {len(peaks)}")
print("First 10 peaks (t, y_s):")
for i in peaks[:10]:
    print(f"t={times[i]:.3f}s  y={y_smooth[i]:.4f}")

cycles = [(times[peaks[k]], times[peaks[k+1]]) for k in range(len(peaks)-1)]
print(f"Total cycles: {len(cycles)}")
print("First 5 cycles (t_start -> t_end):")
for c in cycles[:5]:
    print(c)
"""

# Plot
#plot_results(times_heel, ys_heel, ys_heel_smooth, minimums_heel, 8, TARGET_PATH)
#plot_results
plot_results(times, ys, ys_smooth, peaks, JOINT_IDX, TARGET_PATH)
