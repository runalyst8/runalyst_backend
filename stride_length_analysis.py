import numpy as np
import matplotlib.pyplot as plt
from math_utility import *
from io_utility import *

# Configuration
JSONL_PATH = "run_nlf.jsonl"
TARGET_PATH = "nlf.joints3d[0]"
FOOT_JOINT = "right_heel"


def extract_3d_positions(video_nlf, target_path, joint_idx):
    if isinstance(joint_idx, str):
        joint_idx = JOINT_MAP[joint_idx]

    times, positions = [], []
    for frame_nlf in video_nlf:
        t = frame_nlf.get("timestamp_sec")
        if t is None: continue

        try:
            joints_positions = extract_by_path(nlf_obj=frame_nlf["nlf"], target_path=target_path)
            normalized = normalize_arr(np.array(joints_positions, dtype=np.float32))

            pos = normalized[joint_idx, :]
            positions.append(pos)
            times.append(float(t))
        except:
            continue

    return np.array(times), np.array(positions)


def calculate_stride_lengths():
    video_nlf = open_nlf_output_json(JSONL_PATH)
    times, pos_3d = extract_3d_positions(video_nlf, TARGET_PATH, FOOT_JOINT)

    y_vals = pos_3d[:, 1]
    dts = get_dt(times)
    y_smooth = smooth(y_vals, dts)
    strike_indices = find_minima_indices(y_smooth, dts)

    if len(strike_indices) < 2:
        print("Not enough strides detected.")
        return

    stride_data = []

    for i in range(len(strike_indices) - 1):
        idx_start = strike_indices[i]
        idx_end = strike_indices[i + 1]

        pos_start = pos_3d[idx_start]
        pos_end = pos_3d[idx_end]

        dist = np.linalg.norm(pos_end - pos_start)

        mid_time = (times[idx_start] + times[idx_end]) / 2
        stride_data.append({
            "start_time": times[idx_start],
            "end_time": times[idx_end],
            "mid_time": mid_time,
            "length": dist
        })

    stride_times = [s["mid_time"] for s in stride_data]
    lengths = [s["length"] for s in stride_data]

    plt.figure(figsize=(10, 5))
    plt.step(stride_times, lengths, where='mid', marker='o', linestyle='-', color='b')
    plt.title(f"Stride Length Over Time ({FOOT_JOINT})")
    plt.xlabel("Time (s)")
    plt.ylabel("Stride Length (units)")
    plt.grid(True, alpha=0.3)
    plt.savefig("stride_length_analysis.png")
    plt.show()

    print(f"--- Stride Analysis for {FOOT_JOINT} ---")
    print(f"Total Strides Detected: {len(stride_data)}")
    print(f"Average Stride Length: {np.mean(lengths):.4f}")
    print(f"Standard Deviation: {np.std(lengths):.4f}")

if __name__ == "__main__":
    calculate_stride_lengths()